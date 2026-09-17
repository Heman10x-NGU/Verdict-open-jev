# OpenJev-ModernBERT Failure Gallery & Boundary Audit

Documenting empirical model errors, false acceptances, and boundary edge cases.

| Slice | Case ID | Customer Utterance | Expected | Predicted | Model Confidence |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `missing_option` | `test_mo_00000` | I payed with a card and was charged an extra fee... | `__insufficient_evidence__` | `request_refund` | 32.8% |
| `missing_option` | `test_mo_00001` | I was charged a fee after using my card and I shou... | `__insufficient_evidence__` | `cash_withdrawal_charge` | 42.1% |
| `missing_option` | `test_mo_00002` | Is there a reason I was charged a fee for using my... | `__insufficient_evidence__` | `get_disposable_virtual_card` | 45.8% |
| `missing_option` | `test_mo_00003` | I show another charge on my card from when I used ... | `__insufficient_evidence__` | `automatic_top_up` | 62.7% |
| `missing_option` | `test_mo_00004` | Why do I get charged additional fees on some payme... | `__insufficient_evidence__` | `extra_charge_on_statement` | 93.9% |
| `missing_option` | `test_mo_00005` | I used my card for a purchase and was charged a fe... | `__insufficient_evidence__` | `getting_virtual_card` | 42.6% |
| `missing_option` | `test_mo_00006` | Why was I charged for card payment?... | `__insufficient_evidence__` | `extra_charge_on_statement` | 80.0% |
| `missing_option` | `test_mo_00007` | There is an unauthorized fee.... | `__insufficient_evidence__` | `wrong_exchange_rate_for_cash_withdrawal` | 44.3% |
| `missing_option` | `test_mo_00008` | I would appreciate that someone let me know when t... | `__insufficient_evidence__` | `exchange_charge` | 85.2% |
| `missing_option` | `test_mo_00009` | Why was I charged an extra fee when paying with my... | `__insufficient_evidence__` | `getting_virtual_card` | 34.5% |
| `missing_option` | `test_mo_00010` | Why did I have to pay extra because I paid with ca... | `__insufficient_evidence__` | `card_arrival` | 49.6% |
| `missing_option` | `test_mo_00011` | Why are there fees for card usage?... | `__insufficient_evidence__` | `getting_virtual_card` | 75.3% |
| `missing_option` | `test_mo_00012` | im not sure what this charge is for... | `__insufficient_evidence__` | `exchange_charge` | 84.7% |
| `missing_option` | `test_mo_00013` | What is the fee charged with this card payment?... | `__insufficient_evidence__` | `transfer_fee_charged` | 74.3% |
| `missing_option` | `test_mo_00014` | When would my card be charged an extra fee for a t... | `__insufficient_evidence__` | `visa_or_mastercard` | 49.0% |

### Failure Analysis
1. **Missing-Option Vulnerability**: When the true Banking77 intent is omitted, the model occasionally exhibits high confidence on plausible substitute categories rather than abstaining.
2. **Selective Risk Gating**: Setting an autonomous policy threshold at >=85% safely gates the majority of these errors into the human supervisor review queue.
