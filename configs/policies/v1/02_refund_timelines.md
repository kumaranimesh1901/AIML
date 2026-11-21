# NovaMart Refund Timelines Policy (v1)

## Processing Window After Approval
Once a return is received and inspected, or a refund is approved, NovaMart issues the refund within **2 business days**. If a refund has status `approved` and `refund_issued_at` is not yet set, the payout is still in the 2-business-day processing queue and has not yet been transmitted to the bank.

## Bank Posting Timelines by Payment Method
- **Credit or Debit Card**: 5 to 7 business days after refund issuance.
- **NovaMart Store Credit**: Instant upon refund issuance.
- **Bank Transfer / ACH**: 7 to 10 business days after refund issuance.

## Refund Amount Rules
- Refunds never exceed the original captured payment amount for the returned items.
- Original shipping fees are non-refundable unless the return is due to a NovaMart error (wrong item, damaged item, or carrier delay over 5 business days).
