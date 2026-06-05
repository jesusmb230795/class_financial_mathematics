# Time Value of Money and Fixed Income

This module introduces deterministic cash-flow valuation and fixed-income risk measures.

## Expected outcome

By the end of this module, students should be able to:

- convert cash flows across time using discount factors and interest rates;
- price zero-coupon and coupon bonds;
- distinguish clean price, dirty price, accrued interest, and yield to maturity;
- calculate Macaulay duration, modified duration, dollar duration, and convexity;
- value CETES, Bonos M, and UDIBONOS under simplified Mexican market conventions;
- interpret interest-rate sensitivity and immunization logic.

## Included notes

- time value of money;
- bond pricing, duration, and convexity;
- Interactive Bond Duration-Convexity Dashboard;
- Mexican government bond valuation;
- yield, DV01, and immunization lab.

## Recommended next improvements

- add calendar-aware coupon schedule generation from actual settlement dates;
- connect Banxico UDI and CETES series through the data layer for optional live-data runs;
- extend the dashboard with clean price, dirty price, and accrued interest controls;
- add a liability-driven portfolio optimization exercise after immunization.

## Class sequence

1. Review present value, future value, discount factors, and compounding.
2. Express arbitrary cash-flow streams as discounted sums.
3. Price zero-coupon and coupon bonds under flat yield assumptions.
4. Apply ACT/360 conventions to CETES, Bonos M, and UDIBONOS.
5. Estimate yield to maturity from market price.
6. Measure sensitivity with duration, convexity, and DV01.
7. Use scenario analysis to explain how bond prices react to rate movements.
8. Evaluate a simple Redington immunization setup.
9. Explore the interactive dashboard to compare exact repricing against approximations.

## In-class practice

Students should price a zero-coupon bond, a fixed-rate coupon bond, and a small bond portfolio. They should compare the exact repricing after a yield shock against duration-only and duration-convexity approximations, then explain the clean-versus-dirty price adjustment for a Mexican bond example.

## Module checkpoint

The checkpoint is a bond valuation memo with cash flows, discount assumptions, clean and dirty price, accrued interest, yield, duration, convexity, DV01, and a short immunization interpretation.
