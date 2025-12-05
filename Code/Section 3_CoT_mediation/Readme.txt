README — Reasoning and Emotion Analysis Code
============================================

This folder contains analysis code and data files used for examining
reasoning patterns, emotional content, and mediation effects in this section.
These files correspond to the semantic and reasoning-related analyses
reported in the manuscript.

-----------------------------------------------
1. File List
-----------------------------------------------

(1) CoT_mediation.Rmd
    - R Markdown script containing all analyses related to reasoning-based
      mediation, including:
        • Data import and preprocessing
        • Construction of semantic variables
        • Mediation analyses of reasoning/emotion relations
    - Running this file will reproduce the statistical results and figures
      referenced in the Supplementary Information.

(2) Reasoning_example.xlsx
    - A minimal example dataset illustrating the structure of the reasoning
      annotations used in the study.
    - Columns include:
        • participant/agent ID
        • trial information
        • reasoning text or chain-of-thought

(3) Reasoning_with_emotions.xlsx
    - Dataset containing reasoning content paired with emotional information.
    - Used for semantic categorization analyses and for plotting reasoning-
      emotion relationships.


-----------------------------------------------
2. System Requirements
-----------------------------------------------

Software:
    - R (version 4.2 or later recommended)
    - RStudio (optional)

Required R Packages:
    - tidyverse
    - readxl
    - dplyr
    - ggplot2
    - broom
    - mediation (if used)
    - knitr / rmarkdown (to render Rmd file)

Installation:
    install.packages(c("tidyverse", "readxl", "dplyr", "ggplot2",
                       "broom", "mediation", "knitr", "rmarkdown"))


-----------------------------------------------
3. How to Run the Analysis
-----------------------------------------------

To reproduce all reasoning and mediation analyses:

Step 1:
    Open the file `CoT_mediation.Rmd` in RStudio.

Step 2:
    Ensure that both datasets:
        • Reasoning_example.xlsx
        • Reasoning_with_emotions.xlsx
    are located in the same directory as the Rmd file, or update the file
    paths in the script accordingly.

Step 3:
    Click "Run All" or knit the document:
        knit("CoT_mediation.Rmd")

Step 4:
    Output will include:
        • regression summaries
        • mediation model estimates
        • semantic reasoning plots
        • any figures referenced in the manuscript


-----------------------------------------------
4. Expected Output
-----------------------------------------------

Running the Rmd file should generate:

    - Mediation analysis results examining whether emotional content
      mediates reasoning–decision relationships.
    - Statistical tables exportable to Supplementary Information.





