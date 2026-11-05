Workshop on AI-assisted documentation (explore, plan, build, check with Copilot), then the team clarified its cloud-mask evaluation plan, assigned November tasks, and drafted the presentation hook.

Documentation Workshop
Three documentation layers: docstrings, README (what it does, how to run, results), and honest limits with evidence
Hand-written docs miss details; chat-pasted docs only know pasted context and may invent numbers
Demo used Copilot in a codespace on the hotel cancellations repo to create README.md and briefing.md
Prompt structure: goal, context, rules (do not edit repo substance), success check (every claim auditable)
difference between breifing file and readme file
The README is the polished, public-facing handoff: it explains the project in an easy-to-follow way, including what it is, how to run it, results, and limitations.
The briefing file is for helping you understand the work in more depth—what the AI did, key decisions, technical details, and how the pieces connect.


Cloud Mask Project Plan
Hand-labeled CVAT masks (cloud, clear, uncertain) form the ground truth for evaluating models
Success criteria: at least two classical methods plus one pre-trained model (OmniCloud mask), evaluated against curated reference
Classical methods are rule-based; grayscale brightness thresholding already exists in the Colab
Nurbu Lama will generate base masks with OmniCloud mask then fix edges, as an alternate approach

Task Planning and GitHub Cleanup
Team closed stale October issues and reassigned tasks; Julio Contreras noted seven issues closed this week

Final Presentation
Draft hook: satellite imagery for wildfires and floods is constantly blocked by clouds
Hook must be recorded with everyone present, one person speaking, due Sunday
No labs November 23 to 28; mock presentations resume week of November 30

Next Steps
(Sarrah Abdullahi) Submit November milestone planning, probably on the 7th
(Nurbu Lama) Update the README with team members, tasks, and LinkedIns
(Chidera Okafor) Build Colab cells marking clouds where OmniCloud mask and SAM agree
(Chidera Okafor) Message group chat asking everyone to share links to their work
(pari panchal) Finish manual labeling of remaining images by around November 10 to 12
(Everyone) Record the presentation hook and submit outline by Sunday, November 8
