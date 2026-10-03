# Real public T&C sources — collected so far

Files already in this folder, ready to use (6 documents, 85 clauses):
- `bnpl_lazypay.txt` — LazyPay (BNPL)
- `loan_fibe.txt` — Fibe, formerly EarlySalary (personal loan app)
- `loan_kreditbee.txt` — KreditBee/Krazybee (personal loan app, default/collection clauses)
- `card_hdfc_mitc.txt` — HDFC Bank credit card MITC
- `card_idfc_mitc.txt` — IDFC FIRST Bank credit card MITC
- `insurance_hdfc_ergo.txt` — HDFC ERGO critical illness policy wording

All 4 categories (BNPL, credit card, personal loan, insurance) now have
at least one real document. To reach 15-25 for a stronger dataset, add
more of the same categories — exact sources below.

## BNPL (Buy Now, Pay Later)
- Simpl — search "getsimpl.com terms" (I couldn't pin down the exact live
  URL via search; check the Simpl app's in-app Terms screen directly)
- LazyPay Credit Line (separate doc from the one already saved) — https://lazypay.in/tnc

## Credit cards (MITC = dense, fee/penalty-heavy — great source material)
- ICICI Bank — official welcome-kit page with MITC PDFs per card variant —
  https://www.icici.bank.in/cc-ewelcomekits
- AU Small Finance Bank — official documents page —
  https://www.au.bank.in/cards/credit-card/documents
- Search "[SBI/Axis/Kotak] credit card MITC pdf" for more

## Personal loan apps
- J&K Bank instant e-Loan T&C — https://instantloans.jkb.bank.in/lendperfect/assets/shisu/images/termsConditions.html
- CASHe, MoneyTap, Navi, Dhani — official T&C pages not yet located;
  check each app's in-app terms screen

## Insurance
- HDFC ERGO official policy wordings hub (more policy types) —
  https://www.hdfcergo.com/download/policy-wordings
- Add 2-4 more from other insurers (ICICI Lombard, Bajaj Allianz) by
  searching "[insurer] policy wording pdf"

## Tips for collection
- MITC and policy wording documents are unusually rich for this project
  — dense, specific clause language ("sole discretion", "without notice",
  "final and binding", short claim-filing windows) is exactly what
  categories.py is built to catch.
- Keep a simple spreadsheet mapping filename → product → category → URL
  → date collected, for your report's methodology section.
