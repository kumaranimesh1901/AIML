# NovaMart Duplicate Charges Policy (v1)

## Pending Authorization vs. Captured Duplicate Charge
- **Pending Authorization (`authorised`)**: Temporary holds automatically drop off within **3 to 5 business days** and are not billed.
- **Duplicate Captured Charge (`captured` with `is_duplicate = true`)**: If the customer was billed twice for the same order, NovaMart refunds the duplicate captured amount in full within **3 to 5 business days** after verifying the payment records.

## Agent Verification
Agents must inspect all payment records for the order, confirm whether both charges are `captured` or one is `authorised`, and state the exact duplicate amount without guessing.
