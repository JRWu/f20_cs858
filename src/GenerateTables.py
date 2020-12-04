#!/usr/local/bin/python


# Import DocSim
import sys

sys.path.append('/f20_cs858/src/')
from DocSim import DocSim
import numpy as np



# We collect data for app functionality.
query1 = "We collect the following data."
q1_paragraphs = list([
	"When you access Sleep Cycle, we collect technical information from the devices you use. The information we collect includes IP addresses, timestamps, log files, device type and operating system. We use the information we collect to improve and personalize the Services and to develop new ones. For example, we use the information to troubleshoot and protect against errors; perform data analysis and testing; conduct research and surveys; and develop new features and Services.",
	"Some of this information you provide to us and some we collect when you use our Services. We also may obtain information about you (including personal information) from our business partners, such as vendors, and others.",
	"In order to use the App, we will ask you to enter data directly/indirectly relating to your sleep (including your sleep schedule, your alarm clock time, how do you feel after the sleep, sleeping habits, areas for improving sleep). You will be able to use the App even if you do not give almost all of this data to us. If you do not want to provide us this information, please tap on the next button/arrow on the screens where we ask this data. We also automatically collect from your device language settings, IP address, time zone, type and model of a device, device settings, operating system, Internet service provider, mobile carrier, hardware ID, Facebook ID, and other unique identifiers (such as IDFA and AAID). We need this data to provide our services, analyze how our customers use the app, to serve ads.",
	"What information we collect and how we use it Pillow does not require any form of registration or the creation of a personal user account to use it. You can use Pillow anonymously without having to provide a name, username or e-mail address.",
	"The data read from HealthKit is used solely within the app. It is not collected or disclosed outside of your local installation of the app. Only those data types relevant to the app are accessed. The data read is used to update and verify the sleep data saved to the HealthKit store.",
	"Personalize, Improve, and Develop the Services We use the information we collect to personalize and improve the Services and to develop new ones. For example, we use the information to protect against and troubleshoot errors; conduct data analysis and testing; promote and market the Services; perform surveys and research; and develop new Services and features. When you allow us to collect precise location information, we use that information to provide and improve features of the Services such as providing you community benchmarks in your local area, and verifying your eligibility for Services that require you be within a particular government territory. We may also use your information to show you more relevant content, make inferences, and provide you with personalized insights to help you improve. For example, information such as your height and weight allows us to derive estimates of your resting metabolic rate which helps us to improve the confidence we have in the quality of your sleep results like your estimated total sleep time. As an additional example, information such as your activity, exercise, and sleep data allows us to approximate the effect of changes in your activity or exercise on your ability to improve your sleep.",
	"We collect personal data about you when you give this to us in the course of registering for and/or using our Services for example we may collect your name, address, e-mail address or telephone number.",
	"Use of Your Personal Data General Use. In general, Personal Data you submit to us is used either to respond to requests that you make, or to aid us in serving you better. Data taken from Apple Healthkit is never sold nor shared with 3rd parties or used for advertising purposes. Ways we use your Personal Data include, but is not limited to: (1) to facilitate the creation of and secure your Account on our network; (2) identify you as a user in our system; (3) improve the quality of experience when you use our Applications; (4) send you a welcome email to verify ownership of the email address provided when your Account was created; (5) send you a welcome email to verify ownership of the email address provided when your Account was created; (6) send you administrative email notifications, such as security or support and maintenance advisories; and (7) to send newsletters, surveys, offers, and other promotional materials related to our Applications and for other marketing purposes of Azumio.",
	"If you choose to use our Service, then you agree to the collection and use of information in relation to this policy. The Personal Information that we collect is used for providing and improving the Service. We will not use or share your information with anyone except as described in this Privacy Policy.",
	"Data we collect from you and how we use it SnoreLab only collects data that is relevant to the functionality of the service we provide.",
	"Data We Collect: For users accessing our sleep data services through our app or our other products with location-enhanced features, we request to collect data about your location. This information may include using GPS data, identifying nearby cell towers and Wi-Fi hotspots. Our sleep analysis products necessarily require your location data.",
	"When you access ''ShutEye: Sleep Tracker, Sounds'', we collect technical information from the devices you use. The information we collect includes, but not limited to IP addresses, log files, device type and operating system. We use the information we collect to improve the Services and to develop new features. For example, we use the information to troubleshoot and protect against errors; perform data analysis and testing; conduct research and surveys; and develop new features and Services.",
	"The information that we collect in our Mobile Apps and Websites How we use this information, why we store, and why we retain it How you can request that your information is updated or deleted Important legal and contact information Mobile Apps When you download and use our Mobile Apps, we automatically collect information on the type of device you use, operating system version, country code and language to localize the application and to improve the user experience."])


model_path = '/f20_cs858/models/GoogleNews-vectors-negative300.bin'
from gensim.models.keyedvectors import KeyedVectors
w2v_model = KeyedVectors.load_word2vec_format(model_path, binary=True)
ds = DocSim(w2v_model)

#q1_paragraphs[0]
tokens = list(filter(None,q1_paragraphs))
results = ds.calculate_similarity(q1_paragraphs[0], tokens)




import pandas as pd

q1_table = list()
for par in q1_paragraphs:
	reslist = list()
	results = ds.calculate_similarity(par, q1_paragraphs)
	# Sort by doc to enforce the order
	#results = sorted(results, key = lambda i: i['doc'])
	for q1_score in results:
		reslist.append(q1_score['score'])
	# Append scores for that paragraph
	q1_table.append(reslist)

df = pd.DataFrame(q1_table)







import numpy as np
import matplotlib.pyplot as plt
plt.imshow(np.asmatrix(df))
plt.clim(0,1)
plt.rcParams['image.cmap'] = 'gray'
plt.colorbar()
plt.savefig(fname="matrix.png")
plt.close()
