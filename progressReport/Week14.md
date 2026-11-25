Team will pair each original image, OmniCloudMask prediction and CVAT hand label under matching file names for comparison; mentor wants a story-led presentation hook and side branches merged into main.

Image Comparison Setup:
Each image needs three versions with the exact same name: original, OmniCloudMask prediction, and manual CVAT labeling mask.
Original images failed to load in the notebook because file names did not match and screenshots of overlays were used instead of exported masks.

Mask colors: 
Purple is uncertain, pink is cloud, teal is clear; black is unlabeled area CVAT drawing missed.
Manual labeling covers about 12 images, while the OmniCloudMask batch produced 24, so matching images must be identified.

CVAT Export Process:
Export via Actions, Export task dataset, Segmentation mask format, then take the PNG from the segmentation class folder.
Prefix file name with 'truth mask', upload to shared drive
Steps added to shared doc by Chidera Okafor
CVAT appears to allow only about three free exports, so earlier labels were deleted and must be redrawn.
Labeling split: Chidera Okafor training set, Nina test set, Julio Contreras validation set.

OmniCloudMask Work:
Nurbu found OmniCloudMask is open source and can run locally to batch multiple images; documented in the README.
OmniCloudMask output was poor, so Nurbu hand-corrected masks with a Wacom tablet; his crops differ from the team's.
Sarrah built a Colab step that zips OmniCloudMask outputs (labeled 0 to 24) and original images with matching names.
Original images have large black borders; the cropped images in the Colab adjusted folder should be used instead.

Mentor Feedback on Presentation:
Hook was convoluted; lead with a story (e.g. wildfire rescue team blocked by clouds) explaining why clouds matter.
Technical work praised: ground truth with OmniCloudMask, benchmark notebook, progress report, web app.
Many side branches must eventually be merged into main as project cleanup.
Include demo link plus backup slides with screenshots, and add architecture and data processing pictures.
Do not use industry partner colors without their permission; our team's own cloud-themed palette is fine.

Website and Demo:
Website is hosted on Chidera Okafor's school account; others contribute via GitHub and Chidera Okafor uploads.
Planned additions: manual labeling comparisons, precision/recall section, live demo link plus walkthrough images.
Julio Contreras wants to add content on methods used and how CVAT was used.

Next Steps:
(Pari) Complete validation set CVAT masks by Saturday
(Nurbu) Add methods and CVAT usage content to the website via GitHub
(Sarrah) Rerun OmniCloudMask on cropped images and output originals and predictions with matching names, larger size
(Chidera) Clean up CVAT export instructions in the shared doc
(Chidera) Finish remaining training set masks and upload to the shared drive
(Chidera) Add manual labeling comparisons and precision/recall section to the website
(Pari) Redraw deleted CVAT masks and export as PNG
(Nurbu) Merge team branches into main once Sarrah's work is done
(Nina) Complete test set CVAT masks
Decisions Made
Keep originals, predictions and manual masks in separate folders until done, then combine into one folder
