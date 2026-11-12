Evaluation pipeline, hand-labeled masks and a demo website are nearly done; focus shifts to the November 9 final presentation, with slides in Canva and section owners assigned.

Weekly Progress:
Sarrah finished an evaluation pipeline comparing mask types (precision, recall, F1); it is untested and was waiting on images.
Nurbu extracted images from OCM and studied the classical methods with NotebookLM; tuning them was judged unnecessary.
Plan: run the methods as-is and report if they underperform the pre-trained model
Pari has finished manual ground truth labeling in CVAT: 4 images each for test, training and validation.

Demo Website:
Hugging Face hosting needed a paid plan and Streamlit failed, so Chidera Okafor hosted the prototype on a free school website.
Static version shows 4 images with SAM, Omni Cloud Mask and Consensus masks; live upload version takes about 30 seconds.
Live version only works while Chidera Okafor's laptop runs the backend; README and setup steps are on a branch
Merge timing is unresolved: Sarrah favored a pull request now, Nurbu preferred finishing the website on its branch first then merging later.
Advisor suggested adding view of Pari hand-labeled ground-truth mask alongside the model predictions, and make it a distinct color like blue.
Advisor also suggested adding showing the evaluation results in a table, including accuracy-related scores(recal, f1 score, etc), alongside the live demo.

Final Presentation Planning:
Final presentation is November 9: slides plus a live demo; results tables with accuracy scores recommended, like a research paper.
Completed an intro video which ran about 2 minutes against a 90-second guideline; it is a placeholder and can be remade.
Outline sections: intro/hook, project overview, data understanding and prep, modeling and evaluation, final thoughts.
Group estimates about 15 to 23 minutes per team; actual time slot still to be confirmed
Nurbu suggested writing a concise script with each person's talking points.

Advisor Check-in:
Remaining work is presentation prep and running the pipeline to get precision, recall and F1 stats.
For graphs, advisor recommended borrowing styles from ML papers and blog posts presenting precision and recall.

Next Steps:
(Nurbu) Share NotebookLM notebook so teammates can understand and explain each classical method

(Sarrah) Test the evaluation pipeline on the labeled images and fix issues
(Nina) Work on Hook, Introduction, Project overview slides
(Chidera) Work on data understanding and data preparation slides
(Chidera) Update demo website to show the original image, alongside masks Pari's masks.
(Pari) Work on data understanding and data preperation slides
(Sarrah) Work on Modeling and Evaluation slides
(Everyone) Work on Final thoughts slide.


Decisions Made
Slides will be built in Canva, shared for group editing
Decided on theme for the presentation by group vote
