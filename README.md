# website-classification

## The problem
Problem that we are trying to solve: binary classification of websites using neural networks. The two classes are: dangerous and safe websites.

## Models
We trained Google’s Inception v4 (“Inception”) convolutional neural network and a recurrent neural network with long short-term memory (LSTM) architecture for the sake of solving this problem. For training, we supplied Inception with labeled screenshots of websites and the LSTM with labeled texts displayed on websites.

## Data set for Inception and how it was obtained
We decided to build scraping tools by hand to get the requisite data that we needed to train the algorithms that we did. The challenge of assembling such a dataset was made difficult because half of all the examples that we wanted to scrape were malicious websites that sought to extract information from users, or worse, gain control of the OS. Because of that, we thought it would be wise to use Docker containers on AWS EC2 instances to isolate the resources that were getting us data. These docker containers all pulled from a common queue hosted on AWS SQS and dumped their results into S3 Buckets. This allowed us to get rid of dysfunctional containers running in a docker service quickly after they stalled. 

The data that we extracted from each website was a screenshot and the site’s HTML. We extracted the screenshots by using Selenium driving Firefox and we got the HTML by using Python’s requests library, a dead-simple HTTP interface for the web. We ended up with approximately 13,000 examples, split evenly between malware and non-malware (‘safe’) examples. Our data source for the safe sites was the Alexa’s 1 Million most popular websites and our data source for the malware websites was malwaredomainlist.com. 

## Inception
A description of the Inception’s architecture can be found here. In order to retrain the Inception architecture, we used TensorFlow’s tutorial on how to do so. We also baked some logic into DataSupplier.py (the data in this case being images) to download both classes into a hierarchical structure like the Inception retrainer wanted. 
Downloaded images in iPython: 

>>> from DataSupplier import DataSupplier
>>> cd ~
>>> ds = DataSupplier()

When I tried to do this, a p2.xlarge instance reported a ClientError: An error occurred (403) when calling the HeadObject operation: Forbidden

>>> mkdir classifications
>>> ds.get_all_images(‘classifications’)

Retrained Inception after following instructions in the aforementioned link: 
bazel-bin/tensorflow/examples/image_retraining/retrain --image_dir ~/classifications --how_many_training_steps 16000

To be finished...





