# Analysis Insights  
## Internet Behaviors & Online Shopping Association

This document summarizes the **key insights** derived from the Chi-Square analysis after proper data cleaning and correction of invalid values. It focuses on *what the results mean*, not how the analysis was executed.

---

## Context

The analysis examines the relationship between selected internet behaviors and the likelihood of **Online Shopping** activity.

Chi-Square tests were used to evaluate whether observed differences in shopping behavior across categories were statistically meaningful or likely due to chance.

A critical correction was applied prior to final analysis:  
invalid sentinel values (notably `99` in the Twitter variable) were identified and excluded to prevent distortion of results.

---

## High-Level Findings

After correcting invalid values and re-running the analysis:

- No single internet behavior overwhelmingly explains online shopping behavior
- Some behaviors show **mild association**, but overall effects are modest
- Many features exhibit **weak or statistically insignificant relationships**

This suggests that online shopping behavior is influenced by **multiple interacting factors**, rather than any single internet habit.

---

## Stronger (Relative) Associations

The following attributes demonstrated **higher Chi-Square values relative to others**, indicating a stronger association with Online Shopping *within this dataset*:

- **Years on Internet**  
  Individuals with longer internet experience showed some variation in shopping behavior, suggesting familiarity and comfort with online environments may play a role.

- **Hours Per Day Online**  
  Time spent online showed a mild association, implying that exposure and opportunity may influence shopping likelihood.

These relationships are **associative, not causal**, and should be interpreted cautiously.

---

## Weaker or Non-Significant Associations

Several behaviors showed little to no meaningful relationship with Online Shopping:

- **Twitter Usage**
- **Facebook Usage**
- **Online Gaming**
- **Other Social Networks**
- **Reading News Online**

After correcting invalid values, these features produced:
- Low Chi-Square statistics
- High p-values
- Minimal explanatory value

This indicates that **presence on specific platforms alone does not meaningfully distinguish shoppers from non-shoppers** in this dataset.

---

## Impact of Correcting Invalid Values

Before correction, the presence of `99` in behavioral fields falsely inflated the importance of certain attributes.

After treating these values as *unknown* and excluding them from statistical testing:

- Association strengths decreased
- Rankings shifted
- Results became more consistent with realistic user behavior

This confirms that **data integrity directly affects analytical conclusions**.

---

## Practical Implications

From an applied analytics perspective:

- Internet usage intensity may be a more useful signal than platform choice
- Behavioral data alone may be insufficient for strong segmentation
- Additional variables (e.g., income, purchase history, demographics) would likely be needed for more predictive insight

This reinforces the importance of **feature selection and data enrichment** in behavioral analysis.

---

## Key Takeaway

The most important insight from this analysis is methodological rather than numerical:

> Clean data produces honest results — even when those results are less exciting.

By correcting invalid values and avoiding over-interpretation, this analysis prioritizes **accuracy, transparency, and analytical discipline** over inflated findings.

---

## Next Steps (Optional Extensions)

Future analyses could explore:
- Effect size measures (e.g., Cramér’s V)
- Multivariate models incorporating multiple behaviors
- Comparison of Chi-Square with mutual information
- Behavioral clustering prior to association testing

These steps would deepen understanding while preserving data integrity.
