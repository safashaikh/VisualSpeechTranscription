## Video Demo
A [visual demonstration](https://www.youtube.com/watch?v=ubPe-gHJ0JU "Demo of app recognizing the word "school"") of the application of the Android speech transcription app that was developed.
## Implementation
Our model was trained on the LRW Lip-Reading Dataset published by BBC, which contained cropped videos of people's faces pronouncing words. The dataset can be found [here](http://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html).
### Preprocessing
Our model required a sample of size (29,88,88,1). This means every sample had to be 29 frames of size `88 * 88 * 1`. `88 * 88` represents the frame dimension and `1` represents the grayscale dimension. If we had decided to use color it would be 3, but for the sake of reducing training time we used grayscale.

The BBC Lip-Reading in the Wild (LRW) dataset initially came with 500 word directories with each word including 1000 samples. Each word directory contained a test, train, and val directory with .mp4 files of single word utterances. For scalability, we decided
to train our classifier on only 12 classes, thus choosing only 12 word directories to parse. The 12 word directories were parsed by traversing through each directory for video samples. The video samples were broken up into 29 frames with each frame being center cropped from the original 256×256 to 96×96. The 29 frames were then placed in a directory named after the words where that directory was placed in either a train or test folder depending on which directory the sample was originally from. 

