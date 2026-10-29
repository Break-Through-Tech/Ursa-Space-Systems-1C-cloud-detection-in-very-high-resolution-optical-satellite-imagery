The team clarified the cloud segmentation goal and agreed to refine Omni Cloud Mask ground truth masks for validation and test sets, then raised concerns about low group participation.

Project Goal Recap
The project is semantic segmentation: locate exactly where clouds are in wildfire images, treating smoke and background as ignored.
Smoke only matters so it does not corrupt cloud predictions; the team is not trying to detect smoke.
Ground Truth Masks are the training foundation; Omni Cloud Mask output looked clearly more accurate than Maxar's mask.
Omni Cloud Mask was provided by the CA, John Savage, and is in the Discord under Space Systems 1C.

Refining Ground Truth Masks
Nurbu argued fine-tuning on raw Omni Cloud masks would only match its accuracy, so the masks should be manually improved first.
Images are currently JPEG and should be GeoTIFF, which is sharper (compared to 4K vs 1080p).
Validation set is only about 30 to 40 images, so splitting manual fixes across members is manageable.
Brightness and contrast preprocessing comes later; training images can vary so the model learns stability.

Colab and Tracking Status
Chidera Okafor is running Omni Mask in the Colab against the first image set; test and validation data not yet added.
Wednesday's notebook is on GitHub, and meeting transcripts are logged as GitHub issues for absent members to check off.

Group Participation Concerns
Most members are missing meetings and not reporting on tasks assigned two Wednesdays ago.
If activity stays the same in about three weeks, the team may raise it with Julian and others.
Nurbu will be away next Saturday and Sunday.

Next Steps
(Nurbu) Convert images to GeoTIFF, generate Omni Cloud Ground Truth Masks, upload to Google Drive, and split fixes among members
(Nurbu) Research how to approach the training and fine-tuning step
(Sarrah A Abdullahi) Review Omni Mask from Discord and write a summary, pipeline, and comparison report
(Chidera Okafor) Run Omni Mask on train, test, and validation sets in the Colab and upload the code
(Chidera Okafor) Submit the Discord check-in next Friday so members can add their names
(Chidera Okafor) Ping the group chat about the three-week accountability window
(Chidera Okafor) Log this meeting's transcript as a GitHub issue
(Chidera Okafor) Plans on posting learning videos on ground truth work in the group chat that he finds

Decisions Made
Manually refined Omni Cloud Mask masks will be used for validation and test sets only, not training
Next meeting moved to Thursday since Nurbu is away and Friday is busy
