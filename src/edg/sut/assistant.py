"""Customer-support AI assistant (SUT) with snapshot-isolated tool-calling loop."""

from __future__ import annotations

import json
import time
from typing import Any

from pydantic import BaseModel, Field

from edg.config import SUTVersionConfig
from edg.llm.client import BaseLLMClient, LLMError, _strip_json_fences
from edg.logging import get_logger
from edg.sut.tools import SnapshotToolBox

logger = get_logger(__name__)

_TOOL_INSTRUCTIONS = """You have access to these tools (call at most {max_calls} times):
- get_order(order_id): order status, items, total amount, placement date
- get_shipment(order_id): carrier, expected delivery, scan status, actual delivery, dispute flag
- get_payments(order_id): payment amounts, status, duplicate flag
- get_returns(order_id): return requests, status, refund amount, refund issued date
- search_policy(query): search NovaMart store policies

Respond with ONLY valid JSON in one of two formats:
1) To call a tool: {{"action": "tool_call", "tool": "<tool_name>", "arguments": {{...}}}}
2) To answer the customer (2 to 6 sentences): {{"action": "final_answer", "answer": "<your reply>"}}
"""


# Constrains providers with schema-guided decoding (Ollama `format`) to the two valid reply shapes
SUT_ACTION_JSON_SCHEMA: dict[str, Any] = {
    "anyOf": [
        {
            "type": "object",
            "properties": {
                "action": {"const": "tool_call"},
                "tool": {
                    "enum": ["get_order", "get_shipment", "get_payments", "get_returns", "search_policy"]
                },
                "arguments": {"type": "object"},
            },
            "required": ["action", "tool", "arguments"],
        },
        {
            "type": "object",
            "properties": {"action": {"const": "final_answer"}, "answer": {"type": "string"}},
            "required": ["action", "answer"],
        },
    ]
}


class AssistantResult(BaseModel):
    answer: str
    model_version: str
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    latency_ms: float = 0.0
    error: str | None = None


def _synthesize_grounded_fake_answer(
    *,
    query: str,
    order_id: str,
    toolbox: SnapshotToolBox,
    retrieval_enabled: bool,
) -> str:
    """Deterministic tool-driven answer used only when running with FakeLLMClient."""
    order_info = toolbox.get_order(order_id)
    ship_info = toolbox.get_shipment(order_id)
    pay_info = toolbox.get_payments(order_id)
    ret_info = toolbox.get_returns(order_id)

    policies: list[dict[str, Any]] = []
    if retrieval_enabled:
        pol_res = toolbox.search_policy(query=query, top_k=3)
        policies = pol_res.get("chunks", [])

    order = order_info["order"]
    status = order["status"]
    total = order["total_amount"]
    shipments = ship_info.get("shipments", [])
    payments = pay_info.get("payments", [])
    returns = ret_info.get("refunds_returns", [])

    parts: list[str] = [
        f"I checked your order {order_id}, which currently has status '{status}' and a total of ${total:.2f}."
    ]

    dup_payments = [p for p in payments if p.get("is_duplicate")]
    if dup_payments:
        if retrieval_enabled:
            parts.append(
                f"I see a duplicate captured charge of ${dup_payments[0]['amount']:.2f}; "
                "per our Duplicate Charges Policy, we will refund the duplicate charge in full within 3 to 5 business days."
            )
        else:
            parts.append(
                f"I see two payments on order {order_id}, and our billing team can review the extra charge for you."
            )

    disputed = [s for s in shipments if s.get("customer_disputed")]
    if disputed:
        deliv = (disputed[0].get("actual_delivery") or "")[:10]
        if retrieval_enabled:
            parts.append(
                f"The carrier recorded delivery on {deliv}, and because it is disputed we must complete "
                "a 3 to 5 business day carrier investigation before issuing any refund or replacement."
            )
        else:
            # V1 without policy retrieval occasionally misses the investigation rule
            parts.append(
                f"Although the carrier scan shows delivered on {deliv}, I can go ahead and approve an immediate refund for you."
            )
    elif len(shipments) >= 2:
        delivered_cnt = sum(1 for s in shipments if s.get("actual_delivery"))
        in_transit = [s for s in shipments if not s.get("actual_delivery")]
        exp = (in_transit[0].get("expected_delivery") or "")[:10] if in_transit else "soon"
        parts.append(
            f"Your order shipped in {len(shipments)} separate packages ({delivered_cnt} delivered), "
            f"and the remaining shipment is in transit with expected delivery on {exp}."
        )
    elif shipments:
        s0 = shipments[0]
        if s0.get("actual_delivery"):
            parts.append(
                f"Shipment {s0['shipment_id']} via {s0['carrier']} was delivered on {s0['actual_delivery'][:10]}."
            )
        else:
            parts.append(
                f"Shipment {s0['shipment_id']} via {s0['carrier']} is currently '{s0['carrier_scan_status']}' "
                f"with expected delivery on {s0['expected_delivery'][:10]}."
            )

    if returns:
        r0 = returns[0]
        r_status = r0["status"]
        if r_status == "rejected_past_window":
            if retrieval_enabled:
                parts.append(
                    "Your return request was submitted past our 30-day return window from the delivery date, "
                    "so it is not eligible for a standard refund."
                )
            else:
                parts.append(
                    f"I have reviewed return {r0['return_id']} for ${r0['refund_amount']:.2f} and can process a standard refund for you."
                )
        elif r_status == "approved" and r0.get("refund_issued_at") is None:
            if retrieval_enabled:
                parts.append(
                    f"Your refund of ${r0['refund_amount']:.2f} is approved and in our 2-business-day issuance queue; "
                    "once issued, card refunds post in 5 to 7 business days."
                )
            else:
                parts.append(
                    f"Your refund of ${r0['refund_amount']:.2f} has already been approved and should arrive shortly."
                )
        elif r0.get("reason") == "wrong_item_delivered":
            if retrieval_enabled:
                parts.append(
                    "Because you received the wrong item, we provide a free prepaid return shipping label "
                    "and either a full refund including original shipping or a free replacement."
                )
            else:
                parts.append(
                    f"Return {r0['return_id']} is logged for the wrong item, and you can send it back for a refund."
                )
        else:
            parts.append(
                f"Return {r0['return_id']} has status '{r_status}' for ${r0['refund_amount']:.2f}."
            )

    if "cancel" in query.lower() and status in {"shipped", "out_for_delivery"}:
        if retrieval_enabled:
            parts.append(
                "Because the order has already shipped, it cannot be cancelled in transit; "
                "you may refuse delivery or return it within 30 days of delivery."
            )
        else:
            parts.append("I have submitted a cancellation request for your order.")

    if retrieval_enabled and policies and len(parts) < 3:
        parts.append(f"This follows our {policies[0]['title']}.")

    return " ".join(parts[:5])


class CustomerSupportAssistant:
    """Tool-calling customer-support assistant bound to a single order snapshot."""

    def __init__(
        self,
        *,
        version_config: SUTVersionConfig,
        llm_client: BaseLLMClient,
        max_tool_calls: int = 4,
    ) -> None:
        self.version_config = version_config
        self.llm_client = llm_client
        self.max_tool_calls = max_tool_calls

    def answer(self, *, query: str, order_id: str, toolbox: SnapshotToolBox) -> AssistantResult:
        start = time.perf_counter()

        # Fast deterministic path for FakeLLMClient when no custom handler/responses are scripted
        if (
            self.llm_client.provider == "fake"
            and getattr(self.llm_client, "_handler", None) is None
            and not getattr(self.llm_client, "_responses", None)
        ):
            answer_text = _synthesize_grounded_fake_answer(
                query=query,
                order_id=order_id,
                toolbox=toolbox,
                retrieval_enabled=self.version_config.retrieval_enabled,
            )
            latency_ms = (time.perf_counter() - start) * 1000.0
            return AssistantResult(
                answer=answer_text,
                model_version=self.version_config.name,
                tool_calls=list(toolbox.tool_trace),
                latency_ms=latency_ms,
            )

        system_prompt = (
            f"{self.version_config.system_prompt}\n\n"
            + _TOOL_INSTRUCTIONS.format(max_calls=self.max_tool_calls)
        )
        transcript_parts = [f"Customer query (Order {order_id}): {query}"]

        final_answer: str | None = None
        error_msg: str | None = None

        for step in range(self.max_tool_calls + 1):
            prompt = "\n\n".join(transcript_parts)
            if step == self.max_tool_calls:
                prompt += "\n\nTool call limit reached. Provide your final_answer JSON now."

            try:
                resp = self.llm_client.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    model=self.version_config.base_model,
                    temperature=self.version_config.temperature,
                    json_mode=True,
                    json_schema=SUT_ACTION_JSON_SCHEMA,
                )
            except LLMError as exc:
                error_msg = str(exc)
                logger.error(
                    "sut_llm_error",
                    extra={"order_id": order_id, "version": self.version_config.name, "error": error_msg},
                )
                break

            raw_text = resp.text.strip()
            try:
                parsed = json.loads(_strip_json_fences(raw_text))
            except json.JSONDecodeError:
                # Treat plain-text reply as final answer
                final_answer = raw_text
                break

            action = parsed.get("action")
            if action == "tool_call" and step < self.max_tool_calls:
                tool_name = str(parsed.get("tool", ""))
                args = parsed.get("arguments") or {}
                if not isinstance(args, dict):
                    args = {}
                tool_out = toolbox.dispatch(tool_name, args)
                transcript_parts.append(
                    f"Tool call: {tool_name}({json.dumps(args)})\n"
                    f"Tool result: {json.dumps(tool_out)}"
                )
            elif action == "final_answer" or "answer" in parsed:
                final_answer = str(parsed.get("answer", "")).strip() or raw_text
                break
            else:
                final_answer = raw_text
                break

        if not final_answer:
            final_answer = (
                f"I apologize, I encountered a temporary issue looking up order {order_id}. "
                "Please try again or contact a NovaMart human specialist."
            )

        latency_ms = (time.perf_counter() - start) * 1000.0
        return AssistantResult(
            answer=final_answer,
            model_version=self.version_config.name,
            tool_calls=list(toolbox.tool_trace),
            latency_ms=latency_ms,
            error=error_msg,
        )
