Discord Check In 10/28/2026

ERSA 1C coordinated cloud-mask work: split CVAT ground-truth labeling, got coach feedback on GitHub tracking and model exploration, and the CA advised packaging notebooks into Python files with a cli.py.

Task Check-in and Mask Comparison
Team agreed to review pending tasks and redistribute them after duplicated work and missed estimates during midterms.
Chidera Okafor suggested Sarrah Abdullahi's comparison task add a Colab cell computing cloud coverage % and precision/recall for SAM vs OmniMask.
Builds on Chidera Okafor's demo in the shared Colab precision/recall not done yet
Chidera Okafor cleaned up the shared Colab and keeps a copy on GitHub for coach tracking.

CVAT Ground Truth Labeling
pari panchal demoed browser-based CVAT: auto-segment with visual prompts or bounding box, then fix misses with the brush.
Only 2 to 3 of the 12 images per split (train, validation, test) need annotating, about 6 total.
pari panchal takes validation and test Nina Khilnani and Chidera Okafor assist with the training data set images.

Coach Feedback (Julio)
Posting meeting summaries as GitHub issues is fine a progress report markdown in the repo would be more visible.
Mark issues finished so they do not sit in the Kanban backlog
Validate whether manual fine-tuning improves SAM/OmniMask soon four similar pipelines would call for a standout fifth.

Challenge Advisor Session (John Savage)
No hard rule on labeling detail: ideally cover as much as possible, but rushing for more data is acceptable.
Repos are mostly data and notebooks package code into src/<project> with preprocessing, evaluation and inference files plus a cli.py entry point.
Keep notebooks separate and link the Colab in the README
Push empty files now to set up the structure
Prioritize a simple, cohesive deliverable over depth expect a crunch (90/10 rule).

Next Steps
(Chidera Okafor) Upload today's updated notebook to GitHub at end of session
(Chidera Okafor) Annotate assigned train images in CVAT starting from the bottom
(Chidera Okafor) Work on precision/recall
(pari panchal) *optional* Write a step-by-step CVAT labeling guide for the team if they need more help.
(pari panchal) Annotate the validation and test images in CVAT
(Nina Khilnani) Add remaining Task 1 work to the shared Colab and finish annotating training image

Decisions Made
Ground truth masks use three labels: cloud (pink), clear (teal/green), uncertain (purple) for faint clouds.
