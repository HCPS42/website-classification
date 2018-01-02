# Website classification

## Problem
Using neural networks for classification of websites into two classes: malware and non-malware. 

## Models
We decided to train Google’s Inception v4 (Inception) convolutional neural network and a recurrent neural network with long short-term memory (LSTM) architecture. For training, we supplied Inception and the LSTM with labeled screenshots of websites and texts displayed on them, respectively.

## Data sets
We built scraping tools by hand. Assembling such a dataset was a challenge because half of all the examples that we wanted to scrape were malicious websites that sought to extract information from users, or worse, gain control of the OS. Because of that, we thought it would be wise to use Docker containers on AWS EC2 instances to isolate the resources that were getting us data. These docker containers all pulled from a common queue hosted on AWS SQS and dumped their results into S3 Buckets. This allowed us to get rid of dysfunctional containers running in a docker service quickly after they stalled. 

From each website under consideration, we extracted a screenshot by using [Selenium](http://www.seleniumhq.org/) driving Firefox. We ended up with approximately 13,000 screenshots, split evenly between dangerous and safe examples. Our data source for the safe sites was the Alexa’s 1 Million [most popular websites](https://gist.github.com/chilts/7229605).

The LSTM data set was obtained for the same websites that we used for the Inception dataset. We got the HTML by using Python’s [requests](http://docs.python-requests.org/en/master/) library, a dead-simple HTTP interface for the web. The data source for the malware websites was [malwaredomainlist.com](http://malwaredomainlist.com/). 

Each training example was a sequence of HTML inner text; therefore, we could potentially achieve zero or more training examples from both the malware and safe sites. [Google’s language identifier](https://pypi.python.org/pypi/langdetect?) was used to exclude all languages except for English. This introduced a known error into our data; if we evaluate the model for non-English sites, the classification will be unreliable and not based on any data within the dataset. We wanted to ensure that small strings, such as the inner text of a button, were excluded from the training to prevent overfitting on extraneous signals (‘submit’ as a single-valued HTML inner text could appear in both malware and safe sites), so we excluded all inner text examples under 5 words. You can view code for generating text examples in lstm/preprocess_text.py. 

At the end, there were approximately 20,000 examples, which were split unevenly between the two classes. To address the issue of the imbalance in the LSTM data set, we used two special metrics in addition to the standard accuracy: precision and recall.

## Inception
A description of the Inception’s architecture can be found [here](https://arxiv.org/abs/1602.07261). In order to retrain it, we used [TensorFlow’s tutorial](https://www.tensorflow.org/tutorials/image_retraining) on how to do so. We also baked some logic into DataSupplier.py (the data in this case being images) to download both classes into a hierarchical structure like the Inception retrainer wanted. 

First, we downloaded images in iPython: 

```python
from DataSupplier import DataSupplier
cd ~
ds = DataSupplier()
mkdir classifications
ds.get_all_images(‘classifications’)
```

Next, we retrained Inception after following instructions in the aforementioned link: 
bazel-bin/tensorflow/examples/image_retraining/retrain --image_dir ~/classifications --how_many_training_steps 16000

After training, Inception’s accuracy was about 80%.

## Heatmaps
In light of the surprisingly successful result above, we wanted to understand why the model made the decisions that it did. One way of approximating an explanation that has been proposed in machine learning literature is to generate heatmaps of importance by setting a zero mask over the majority of input data except a small section for which we are measuring importance. The window slides to measure importance over the entire image; the softmax score is taken for every masked version of the image as the tile of non-zero values slides over it and all the evaluations give us a good idea of which part of the image resulted in the overall classification.

Malware-classified images seemed to have hot spots consistently over large white-space areas and therefore we can hypothesize that large swathes of white space are highly indicative of suspicious and dangerous sites. On the other hand, safely-classified sites generally have little white space. This intuitively makes sense because the training set that we used consisted of the most popular pages on the web (as determined by a website’s Alexa score). These websites are often run for-profit and try to maximize engagement, and using all the screen real estate available likely maximizes engagement. 

You can view the code for generating heatmaps in cnn/generate_heatmaps.py.

## LSTM 
An explanation of how LSTM works can be found [here](http://colah.github.io/posts/2015-08-Understanding-LSTMs/). The actual architecture that we used can be found in lstm/LSTM.py. After training, the LSTM’s accuracy was around 86%, and the precision and recall were about 0.7. 

The code for reusing the LSTM is in lstm/using_saved_model.py.

## Conclusion
The results we achieved are surprising and fascinating — they showed that there are in fact visual signals simply in the layout of sites that indicate whether or not they are malicious.



