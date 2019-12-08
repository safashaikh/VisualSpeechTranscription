## Video Demo
A [visual demonstration](https://www.youtube.com/watch?v=ubPe-gHJ0JU "Demo of app recognizing the word "school"") of the application of the Android speech transcription app that was developed.
## Implementation
Our model was trained on the LRW Lip-Reading Dataset published by BBC, which contained cropped videos of people's faces pronouncing words. The dataset can be found [here](http://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html).
### Preprocessing
Our model required a sample of size (29,88,88,1). This means every sample had to be 29 frames of size `88 * 88 * 1`. `88 * 88` represents the frame dimension and `1` represents the grayscale dimension. If we had decided to use color it would be 3, but for the sake of reducing training time we used grayscale.
