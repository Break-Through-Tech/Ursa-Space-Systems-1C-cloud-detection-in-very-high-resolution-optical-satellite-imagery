Team 1C (Ursa cloud segmentation) shared progress, got coach and challenge advisor feedback on GitHub hygiene, fine-tuning, research sources and ground truth masks, then visited other teams and joined the shoutout trivia.

Team Progress and Coach Feedback
Project goal: compare a from-scratch model with a fine-tuned Hugging Face model, starting with fine-tuning
Off-the-shelf segmentation estimated 82 to 88% cloud cover on an image with about 10%; kept as a baseline
Coach: fine-tune SAM on a small subset (about 8 tiles); SAM itself is optional
Coach flagged last GitHub push was September 13th; push notebooks periodically to the Notebooks folder for auditability and interviews
Dataset expanded from one of four sections (about 25 images, overfit risk) to all four: two train, one validation, one test, tracked in a CSV

Research Direction and Milestones
October milestone: survey models on Hugging Face plus IEEE papers and conferences like CVPR, ICCV, NeurIPS (ECCV heard as 'ECCB')
Big-player foundation models (Google, Meta) fine-tune easily; comparing them to cloud-only models mirrors real research
Coach office hours: Friday 1:00 p.m. Eastern Time, booked through the Discord link

Challenge Advisor Session
Advisor is building a Python package that generates ground truth cloud masks from image URLs as 1/0 arrays at full image size
Pre-trained model scored 95 to 98 against rare existing ground truth masks
Teams must chip masks the same way they chip images
Suggested arXiv and Google Scholar for papers; search term to use is 'segmentation models' (binary segmentation fits)
Another team used AI to generate next milestones from their progress; Coach Jullio liked the idea

Cross-Team Visits and Program Wrap-Up
Nurbu and Chidera visited other teams and learned about what their projects were doing
Shoutouts ran as a live trivia; assignments due November 1st; Friday Discord check-ins and event survey reminded

Next Steps
(Chidera Okafor) Push the Colab notebook to the GitHub Notebooks folder by end of day
(Coach Jullio) Share list of conferences and papers on cloud segmentation approaches
(Chidera) Propose using AI to generate milestones to the team
(Chidera) Clean up the Whisper transcript and upload it as a GitHub issue with read checkmarks
(Chidera) Message the team Friday to schedule the weekly meeting
(Challenge Advisor Jack Savage) Post the ground truth mask package, starter Colab and GitHub README on Discord
(Our Team) Run the advisor's mask code and report any issues back

Decisions Made
Weekly team meetings move to Sunday evenings after 6:00
Meeting transcripts go into GitHub issues with per-member read checkmarks
