#!/usr/local/bin/python


# Import DocSim
import sys

sys.path.append('/f20_cs858/src/')
from DocSim import DocSim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the Word2Vec Model from Google
model_path = '/f20_cs858/models/GoogleNews-vectors-negative300.bin'
from gensim.models.keyedvectors import KeyedVectors
w2v_model = KeyedVectors.load_word2vec_format(model_path, binary=True)
ds = DocSim(w2v_model)


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

query2 = "We share your data with third parties."
q2_paragraphs = list([
"We may anonymize, de-identify and/or aggregate the information that we collect and use such anonymized, de-identified and/or aggregated data for commercial, statistical and market research purposes, including sharing it with affiliates and business partners. You own your personal data, and you’re always in control. Therefore we offer you several ways to control the privacy of your personal information and we’re constantly striving to improve the functions and features needed in order for you to be feel that you’re in control. Our privacy policy (the “Privacy Policy”) is here to give you all the information you need about what kind of information we collect, how we use, how we share it, how you control your data, and make sure you know your rights in connection to when you use our applications, software, websites, APIs and services (collectively, “Sleep Cycle”).",
"Marketing providers. We, or the third-party service providers we use to assist us with marketing our own products to you, may use the information we collect from you to provide advertisements and offers for our other products. Additionally, we may share certain information with Facebook that allows us to create Custom or Lookalike Audiences. You may learn more about Facebook Lookalike Audiences click here and about your off-Facebook activity and how to opt out of having such activity sent to Facebook click here. We encourage you to review Facebook's Privacy Policy here. Additionally, if we obtain your information in connection with a contest, sweepstakes, event, offering or other promotional activity that is jointly offered by us and any third parties. By entering such contest or sweepstakes, you authorize and direct us to share your information with our co-sponsor. We may also share your information with a third-party service provider who administers the promotion, contest and/or sweepstakes. We may also share aggregate information to third party providers and platforms that help us understand our user demographic, including user demographic interests, habits and usage patterns for certain of our Services so that we may market our products more effectively.",
"For improving the app and serving ads, we may share this data with third parties. Such parties include Facebook, Google, Apple, Appsflyer, Amplitude, Snapchat, Crashlytics, Firebase, Fabric. As a result of sharing this data with third parties we (1) analyze different interactions (how often users make subscriptions, how many users chose a particular area for improving sleep or have similar sleep schedule); (2) serve ads (and are able to show them only to a particular group users, for example, to subscribers).",
"Any data processing of your health data will take place exclusively on the devices that you use to run Pillow. Pillow will not store or cache any of your health data that it will access from Apple Health and it will not transmit this data to external servers either owned by Neybox Digital Ltd. or any other third party. We will not sell your Health Data to advertising platforms, affiliate networks, data brokers or information resellers. We will not use your Health Data for advertising, promotion, cross-selling or any similar service.",
"In no case is your personal data ever transfered to a 3rd party. It is only ever kept within the application container of Sleep++ or transfered to your personal HealthKit data store as per your permission choices.",
"We may de-identify or aggregate the information you make available in connection with the Services, in ways that do not personally identify you. Examples of such aggregated information or statistical data include the average total sleep time and average sleeping heart rate of the SleepWatch community. We may use, sell, license, and share this de-identified or aggregated information with third parties for research, business or other purposes such as to help advance scientific understanding of sleep and help advance scientific understanding of ways to improve sleep. Please visit your account settings if you object to SleepWatch using your information for these purposes.",
"When you use any of our Services, we may disclose your personal data to the following parties: To our group companies, third party services providers and partners who provide data processing services to us. For example, to support the delivery of, provide functionality on, or help to enhance the security of our Services, or who otherwise process personal data for purposes that are described in this Privacy Policy or notified to you when we collect your personal data. To any competent law enforcement body, regulatory, government agency, court or other third party where we believe disclosure is necessary. To any other person with your consent to the disclosure.",
"Creation of Anonymous Data. We may create Anonymous Data records from Personal Data by excluding information (such as your name) that make the data personally identifiable to you. We use this Anonymous Data to analyze request and usage patterns so that we may enhance the content of our Applications. Azumio reserves the right to use Anonymous Data for any purpose and disclose Anonymous Data to third parties, including but not limited to our research partners, in its sole discretion. This does not include genetic data.",
"The app does use third party services that may collect information used to identify you.",
"We will not distribute any of your personal data to third parties, except if it is required to provide the service to you (e.g. technical service providers as detailed in subsequent sections), unless we have asked for your explicit consent.",
"The prime purpose of collecting and using your data is to allow us to properly measure and create a personalized path to achieve better sleep. The use of your collected personal sleep data is necessary for SleepScore to deliver Services that involve analyzing and evaluating Sleep conditions and solutions. We do not and will not sell your data to third parties.",
"We work with partners who provide us with analytics services. This includes helping us understand how users interact with the ''ShutEye: Sleep Tracker, Sounds'' services, the stability, and performance of the services. These companies may use device-specific advertiser identifiers, cookies and similar technologies to collect information about your interactions with the Services and other websites and applications.",
"If you do not consent to a personalized advertising experience, or you later withdraw consent, our ad mediation partner may still process your personal data when necessary for fraud detection or to comply with the law."])
# pillow and Sleep++ claim to not share with 3rd parties

query3 = "We will delete all of your data upon request."
q3_paragraphs = list([
	"If you wish to cancel your account or request that we no longer use your personal information to provide you the Service, you may delete your account by sending a request to delete your account to support@sleepcycle.com. We may also delete your account if it has been inactive for a certain amount of time. Preceding the delete request it may take up to 30 days to fully delete your personal information from our systems.",
	"Deletion: in certain circumstances, you can request a right to be forgotten (this is a right to have your information deleted or our use of your data restricted). We will honor such requests unless we have to retain this information to comply with a legal obligation or unless we have an overriding interest to retain it;",
	"When you request deletion of your personal data, we will use reasonable efforts to honor your request. In some cases we may be legally required to keep some of the data for a certain time; in such event, we will fulfill your request after we have complied with our obligations.",
	"Choose to send a request to delete all email messages and data that you have exchanged with your Helpdesk, by contacting us via email. Choose to send a request to delete any anonymous data gathered by the third-party analytics services that we use that could be tied indirectly to your device to the degree that’s possible since for those services to isolate data tied to your particular device.",
	"If you have questions about deleting or correcting your personal data please contact our support team.",
	"We retain information as long as it is needed to provide the Services to you and others, subject to any legal obligations to further retain such information. We keep your account information, like your email address and password, for as long as your account is in existence because we need it to operate your account. We keep other information, like your heart rate, exercise or activity data, until you use your account settings or tools to delete the data or your account because we use this data to provide you with your personal statistics and other aspects of the Services. We also keep information about you and your use of the Services for as long as necessary for our legitimate business interests, for legal reasons, and to prevent harm, including as described in the sections How We Share Information and How We Use Information. Following your deletion of your account, while most of your information will be deleted within 30 days, it may take up to 90 days to fully delete your personal information from our systems. Additionally, we may retain information from deleted accounts to comply with the law, prevent fraud, collect fees, resolve disputes, troubleshoot problems, assist with investigations, enforce the Terms of Service and take other actions permitted by law. The information we retain will be handled in accordance with this Privacy Policy. Information about you that is no longer necessary and relevant to provide our Services may be de-identified and aggregated with other non-personal data to provide insights which are commercially valuable to us, such as statistics of the use of the Services and community benchmarks of health information.",
	"Unless a longer retention period is required or permitted by law, we will only retain your personal data only for as long as reasonably necessary to fulfil the purposes outlined in this Privacy Policy and for our legitimate business interests, such as to comply with our legal obligations, resolve disputes, and enforce our agreements. We will for example periodically de-identify unused user accounts and regularly review our data sets.",
	"We will retain User Provided Data for as long as you use the Applications and for a reasonable time thereafter. If you would like us to delete User Provided Data, please contact us at support@azumio.com and we will respond within a reasonable time. Please note that some or all of the User Provided Data may be required in order for the Applications to function properly, and we may be required to retain certain information by law. We offer you choices regarding the collection, use, and sharing of your Personal Data.",
	"For a better experience, while using our Service, we may require you to provide us with certain personally identifiable information, including but not limited to email, Apple Health data. The information that we request will be retained by us and used as described in this privacy policy.",
	"We may retain certain personal information in an aggregated and anonymized format after your account has been deleted. We reserve the right to use your information in any aggregated form after you have deleted your account, but will ensure that the use of this information will not personally identify you.",
	"You may request that your account is deleted by contacting the company at support@sleepscorelabs.com or by phone at (858) 299-8995. Once deleted, your data, including your account, activities and place on leaderboards cannot be reinstated.",
	"You can request the deletion of your data by contacting us at contact.sleep@enerjoy.life. After that, you can use our application as usual, without any discrimination. ",
	"Our Websites contain functionality to allow you to optionally contact us via email. When received, we retain submitted email addresses and any submitted personal information. No information is shared with any third party, or used for any other purpose. You can always send us an email to contact@phase4mobile.com if you’d like to remove any of your submitted information."])


query4 = "We will retain your data after deletion."
q4_paragraphs = list([
	"We keep your account information, such as your email address, and password, for as long as your account is in existence because we need it to operate your account. We keep other information, like your activity data, until you use your tools to delete the data or contact us for a full erasure, because we use this data to provide you with your personal statistics and other aspects of the Services. We also keep information about you and your use of the Services for as long as necessary for our legitimate business interests, for legal reasons, and to prevent harm.",
	"Except as provided below, we may retain your personal information for the longer of three (3) years after we become aware that you have ceased using our services or for so long as necessary to fulfil any contractual obligation governing the information or our legal and regulatory obligations. However, we may not know if you have stopped using our services, so we encourage you to contact us at the appropriate support email in the How to Contact Us section of this Privacy Policy if you are no longer using the services. We may retain other information that is not personally identifiable for backups, archiving, prevention of fraud and abuse, analytics, or where we otherwise reasonably believe that we have a legitimate reason to do so.",
	"We will store your personal data for as long as it is reasonably necessary for achieving the purposes set forth in this Privacy Policy (including providing the Service to you), which includes (but is not limited to) the period during which you have an account with the App. We will also retain and use your personal data as necessary to comply with our legal obligations, resolve disputes, and enforce our agreements.",
#	"NOT AVAILABLE",
#	"NOT AVAILABLE",
	"We retain information as long as it is needed to provide the Services to you and others, subject to any legal obligations to further retain such information. We keep your account information, like your email address and password, for as long as your account is in existence because we need it to operate your account. We keep other information, like your heart rate, exercise or activity data, until you use your account settings or tools to delete the data or your account because we use this data to provide you with your personal statistics and other aspects of the Services. We also keep information about you and your use of the Services for as long as necessary for our legitimate business interests, for legal reasons, and to prevent harm, including as described in the sections How We Share Information and How We Use Information. Following your deletion of your account, while most of your information will be deleted within 30 days, it may take up to 90 days to fully delete your personal information from our systems. Additionally, we may retain information from deleted accounts to comply with the law, prevent fraud, collect fees, resolve disputes, troubleshoot problems, assist with investigations, enforce the Terms of Service and take other actions permitted by law. The information we retain will be handled in accordance with this Privacy Policy. Information about you that is no longer necessary and relevant to provide our Services may be de-identified and aggregated with other non-personal data to provide insights which are commercially valuable to us, such as statistics of the use of the Services and community benchmarks of health information.",
	"Unless a longer retention period is required or permitted by law, we will only retain your personal data only for as long as reasonably necessary to fulfil the purposes outlined in this Privacy Policy and for our legitimate business interests, such as to comply with our legal obligations, resolve disputes, and enforce our agreements. We will for example periodically de-identify unused user accounts and regularly review our data sets.",
	"We will retain User Provided Data for as long as you use the Applications and for a reasonable time thereafter.",
#	"NOT AVAILABLE",
	"We may retain certain personal information in an aggregated and anonymized format after your account has been deleted. We reserve the right to use your information in any aggregated form after you have deleted your account, but will ensure that the use of this information will not personally identify you.",
	"We retain information as long as it is necessary to provide the Services to you and others, subject to any legal obligations to further retain such information. Information associated with your account will generally be kept until it is no longer necessary to provide the Services or until you ask us to delete it or your account is deleted whichever comes first. For example, when you withdraw your consent for SleepScore Labs to process your health-related information, SleepScore Labs will delete all health-related information you uploaded. Following the deletion of your account, it may take up to 30 days to fully delete your personal information and system logs from our systems. Additionally, we may retain information from deleted accounts to comply with the law, prevent fraud, collect fees, resolve disputes, troubleshoot problems, assist with investigations, enforce the Terms of Service and take other actions permitted by law. The information we retain will be handled in accordance with this Privacy Policy.",
#	"NOT AVAILABLE",
#	"NOT AVAILABLE",
	])










# Plot Q1 Comparison Matrix
tokens = list(filter(None,q1_paragraphs))
results = ds.calculate_similarity(q1_paragraphs[0], tokens)

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

q1_df = pd.DataFrame(q1_table)

plt.imshow(np.asmatrix(q1_df))
plt.clim(0,1)
plt.rcParams['image.cmap'] = 'gray'
plt.colorbar()
plt.title("Query 1 Comparison Matrix")
plt.savefig(fname="q1_comparison_matrix.png")
plt.close()


# Plot Q2 Comparison Matrix
tokens = list(filter(None,q2_paragraphs))
results = ds.calculate_similarity(q2_paragraphs[0], tokens)

q2_table = list()
for par in q2_paragraphs:
	reslist = list()
	results = ds.calculate_similarity(par, q2_paragraphs)
	# Sort by doc to enforce the order
	#results = sorted(results, key = lambda i: i['doc'])
	for q2_score in results:
		reslist.append(q2_score['score'])
	# Append scores for that paragraph
	q2_table.append(reslist)

q2_df = pd.DataFrame(q2_table)

plt.imshow(np.asmatrix(q2_df))
plt.clim(0,1)
plt.rcParams['image.cmap'] = 'gray'
plt.colorbar()
plt.title("Query 2 Comparison Matrix")
plt.savefig(fname="q2_comparison_matrix.png")
plt.close()


# Plot Q3 Comparison Matrix
# Min 0.744749
tokens = list(filter(None,q3_paragraphs))
results = ds.calculate_similarity(q3_paragraphs[0], tokens)

q3_table = list()
for par in q3_paragraphs:
	reslist = list()
	results = ds.calculate_similarity(par, q3_paragraphs)
	# Sort by doc to enforce the order
	#results = sorted(results, key = lambda i: i['doc'])
	for q3_score in results:
		reslist.append(q3_score['score'])
	# Append scores for that paragraph
	q3_table.append(reslist)

q3_df = pd.DataFrame(q3_table)

plt.imshow(np.asmatrix(q3_df))
plt.clim(0,1)
plt.rcParams['image.cmap'] = 'gray'
plt.colorbar()
plt.title("Query 3 Comparison Matrix")
plt.savefig(fname="q3_comparison_matrix.png")
plt.close()


# Plot Q4 Comparison Matrix
tokens = list(filter(None,q4_paragraphs))
results = ds.calculate_similarity(q4_paragraphs[0], tokens)

q4_table = list()
for par in q4_paragraphs:
	reslist = list()
	results = ds.calculate_similarity(par, q4_paragraphs)
	# Sort by doc to enforce the order
	#results = sorted(results, key = lambda i: i['doc'])
	for q4_score in results:
		reslist.append(q4_score['score'])
	# Append scores for that paragraph
	q4_table.append(reslist)

q4_df = pd.DataFrame(q4_table)

plt.imshow(np.asmatrix(q4_df))
plt.clim(0,1)
plt.rcParams['image.cmap'] = 'gray'
plt.colorbar()
plt.title("Query 4 Comparison Matrix")
plt.savefig(fname="q4_comparison_matrix.png")
plt.close()

# The black ones are due to the information "not available".

np.min(np.min(q1_df))
# 0.7698
np.min(np.min(q2_df))
#0.6895

np.min(np.min(q3_df))
#0.7447

np.min(np.min(q4_df))
#0.5219

# Those that didn't exist were excluded from our result(s)


q1_fitness = list([
	"Under Armour collects, uses, discloses and processes Personal Data as outlined in this Privacy Policy, including to operate and improve the Services and our business; for advertising and marketing; and to provide you with innovative fitness and wellness services, as further described in this Privacy Policy.",
	"Using the information we collect, we are able to deliver the Services to you and honor our Terms of Service contract with you. For example, we need to use your information to provide you with your Fitbit dashboard tracking your exercise, activity, and other trends; to enable the community features of the Services; and to give you customer support.",
	"Unless specified otherwise, all Data requested by this Application is mandatory and failure to provide this Data may make it impossible for this Application to provide its services. In cases where this Application specifically states that some Data is not mandatory, Users are free not to communicate this Data without consequences to the availability or the functioning of the Service."])

q1_diet = list([
	"As you continue to use the Services, you will regularly provide Lifesum with further personal data. It follows from the nature of the Services that we must process such data that you upload to the Services to enable the Services, for example, we will process your weight data and calorie intake to enable the monitoring and presentation of your personal goals (whether it be weight loss or weight gain). This processing is a pre-requisite for us being able to offer the Services to you.",
	"The subject of data protection is very close to the heart of the Fastic GmbH and therefore we would like to make it as transparent as possible to the user, how and for what purpose his data will be used. For example, some information is required to provide the user with personalized functions and content in FasticApp or on other related platforms or to provide the user with suitable offers around the FasticApp services (e.g. notes on additional content, special offers as well as discounts for the FasticApp services). The data of the user will of course be handled responsibly and will only be used within the framework of the applicable data protection laws, in particular the EU Data Protection Basic Regulation (EU-DSGVO).",
	"Personal Information. Your privacy is important to Noom, and Noom is committed to carefully managing your individually identifiable information (“Personal Information”) in connection with the Services that Noom provides. “Personal Information” means any information that may be used, either alone or in combination with other information, to personally identify an individual, including, but not limited to, a first and last name, a personal profile, an email address or other contact information. This Privacy Policy describes the information practices for Noom, including what type of information is gathered and tracked, how the information is used, and with whom the information is shared. Noom is committed to protecting the privacy of the data you provide in the Service as appropriate, but at the same time encouraging you to interact with and share information about you progress with other users using the Services."])

q2_fitness = list([
	"We may allow you to register and pay for third-party products and services or otherwise interact with another website, mobile application, or Internet location (collectively Third Party Sites) through our Services, and we may collect Personal Data that you share with Third Party Sites through our Services. When we do so, we will inform you of the further details of how we use your Personal Data.",
	"We may share non-personal information that is aggregated or de-identified so that it cannot reasonably be used to identify an individual. We may disclose such information publicly and to third parties, for example, in public reports about exercise and activity, to partners under agreement with us, or as part of the community benchmarking information we provide to users of our subscription services.",
	"Users are responsible for any third-party Personal Data obtained, published or shared through this Application and confirm that they have the third party's consent to provide the Data to the Owner."
])

q2_diet = list([
	"Personal data collected from you may be shared with third-party providers of Lifesum that process personal data on behalf of Lifesum; such as server hosting providers, data storage providers, companies carrying out system and sales performance monitoring, customer support systems- and payment service providers. These service providers will be considered processors of your personal data.",
	"FasticApp and analysis service providers of the FasticApp Service may analyze activity data for research purposes designed to provide personalized service and promote healthy habits. FasticApp may share user data obtained through the HealthKit framework or the Google Fit SDK with a third party for medical research with the express consent of the user. The FasticApp service will not use information obtained through HealthKit or Google Fit SDK applications for advertising or similar services. The user may prevent the FasticApp service from accessing his or her data at any time by changing the settings of his or her mobile device. Anyone using HealthKit or Google Fit SDK to store and analyze their sensitive data should take care to protect their smartphone with a secure code (e.g., on the iPhone under Touch ID & Code, disable the simple code and create a password using a combination of uppercase, lowercase, numbers and special characters).",
	"The Website, Mobile App and Services may include functionality that allows certain kinds of interactions between the Website, Mobile App and Services and User’s account on a third-party web site or application. The use of this functionality may involve the third-party operator providing certain information, including Personal Information, to Noom. For example, when User registers with the Website and/or Mobile App, User may have an option to use User’s Facebook, Google or other account provided by a third-party site or application to facilitate the registration and log-in or transaction process on the Website, Mobile App and Services or otherwise link accounts. If Noom offers and User chooses to use this functionality to access Noom’s Website, Mobile App and Services, the third-party site or application may send Personal Information about User to the Website and/or Mobile App. If so, Noom will then treat it as Personal Information under this Privacy Policy, since Noom is collecting it as a result of User’s accessing of and interaction on Noom’s Website, Mobile App and Services. In addition, Noom may provide third-party sites’ interfaces or links on the Website, Mobile App and Services to facilitate User’s sending a communication from the Website, Mobile App and Services. For example, Noom may use third parties to facilitate emails, text messages, blog postings, tweets or Facebook postings. These third parties may retain any information used or provided in any such communications or other activities and these third parties’ practices are not subject to Noom’s Privacy Policy. Noom may not control or have access to User’s communications through these third parties. Further, when User use third-party sites or services, User is using their services and not Noom’s services and they, not Noom, are responsible for their practices. User should review the applicable third-party privacy policies before using such third-party tools on Noom’s Website."
])

q3_fitness = list([
	"Where permissible, we will also delete your Personal Data upon your request. Information on how to make a deletion request can be found here.",
	"If you choose to delete your account, please note that while most of your information will be deleted within 30 days, it may take up to 90 days to delete all of your information, like the data recorded by your Fitbit device and other data stored in our backup systems. This is due to the size and complexity of the systems we use to store data. We may also preserve data for legal reasons or to prevent harm, including as described in the How Information Is Shared section.",
	"Have their Personal Data deleted or otherwise removed. Users have the right, under certain circumstances, to obtain the erasure of their Data from the Owner."
])

q3_diet = list([
	"By submitting User Material to Lifesum, you warrant and represent that you hold the copyright, trademark and/or other intellectual property rights to your content. You agree to grant Lifesum a non-exclusive, transferable, sub-licensable, royalty-free, worldwide license to use User Material to the extent necessary for Lifesum to operate and maintain the Service. This license shall remain valid until the respective User Material is deleted from the Service by you or by Lifesum in accordance with these Terms.",
	"Revocation / Opt-out possibility: The user has the possibility to delete his profile and all personal data stored therein at any time by sending his revocation to datenschutz@getfastic.com. The provider will then forward this revocation to GF, who have undertaken to delete the corresponding data. Furthermore, the provider will also delete the user’s account if the user does not actively use any of our FasticApp services for a period of three years. If and to the extent that the data associated with the user’s account can and must still be used for purposes which have not yet ceased to exist at the time of the desired or planned deletion, the data records will at least be blocked or limited to certain processing purposes instead of being deleted. This is particularly the case in the case of legally mandatory storage obligations such as the corresponding commercial and tax law regulations. The latter can be up to 10 years (see § 147 (3) of the German Fiscal Code).",
	"Access/Accuracy. To the extent that you do provide us with Personal Information, we wish to maintain accurate Personal Information. If you would like to delete or correct any other of your Personal Information that we may be storing, you may submit a request to us by sending an email to support@noom.com. Your email should include adequate details of your request."])

q4_fitness = list([
	"We will retain your Personal Data for as long as you maintain an account or as otherwise necessary to provide you the Services. We will also retain your Personal Data as necessary to comply with our legal obligations, resolve disputes, and enforce our agreements. Where we no longer need to process your Personal Data for the purposes set out in this Privacy Policy, we will delete your Personal Data from our systems.",
	"If you choose to delete your account, please note that while most of your information will be deleted within 30 days, it may take up to 90 days to delete all of your information, like the data recorded by your Fitbit device and other data stored in our backup systems. This is due to the size and complexity of the systems we use to store data. We may also preserve data for legal reasons or to prevent harm, including as described in the How Information Is Shared section.",
	"The Owner may be allowed to retain Personal Data for a longer period whenever the User has given consent to such processing, as long as such consent is not withdrawn. Furthermore, the Owner may be obliged to retain Personal Data for a longer period whenever required to do so for the performance of a legal obligation or upon order of an authority."])

q4_diet = list([
#	"",
	"The user data (e-mail address, name and user name) will be deleted from the provider’s system after one year and one month at the latest. In the case of deletion requests for the newsletter, a connection to the user’s user account can be established using the provider’s own system, provided that the user’s registration address is involved. For requests to delete a user account, no connection can be established to the user’s account. The data is stored in the system protected against unauthorized access and will not be passed on to third parties.",
	"Requests to delete Personal Information are subject to any applicable legal and ethical reporting or document retention obligations imposed on Noom."])

q1_all = q1_paragraphs + q1_fitness + q1_diet
q2_all = q2_paragraphs + q2_fitness + q2_diet
q3_all = q3_paragraphs + q3_fitness + q3_diet
q4_all = q4_paragraphs + q4_fitness + q4_diet

all_sets = list(["Query 1", "Query 2", "Query 3", "Query 4"])
all_queries = list([q1_all, q2_all, q3_all, q4_all])

for idx, q in enumerate(all_queries):
	#print(idx)
	#print(q)
	#tokens = list(filter(None,all_queries[idx]))
	result_table = list()
	for par in all_queries[idx]:
		reslist = list()
		results = ds.calculate_similarity(par, all_queries[idx])
		for q_score in results:
			reslist.append(q_score['score'])
		result_table.append(reslist)
		q_df = pd.DataFrame(result_table)
	print("Query: " + str(idx)+ " " + str(np.min(np.min(q_df))))
	plt.imshow(np.asmatrix(q_df))
	plt.clim(0,1)
	plt.rcParams['image.cmap'] = 'gray'
	plt.colorbar()
	plt.title(all_sets[idx] + " Comparison Matrix")
	plt.savefig(fname="rq2_q" + str(idx) + "_comparison_matrix.png")
	plt.close()


"""<matplotlib.image.AxesImage object at 0x7f5e0243fbe0>
<matplotlib.colorbar.Colorbar object at 0x7f5e0243fe80>
Text(0.5, 1.0, 'Query 1 Comparison Matrix')
Query: 1 0.6895198225975037
<matplotlib.image.AxesImage object at 0x7f5e023dbdf0>
<matplotlib.colorbar.Colorbar object at 0x7f5e023dbdc0>
Text(0.5, 1.0, 'Query 2 Comparison Matrix')
Query: 2 0.726014256477356
<matplotlib.image.AxesImage object at 0x7f5e02307070>
<matplotlib.colorbar.Colorbar object at 0x7f5e02307310>
Text(0.5, 1.0, 'Query 3 Comparison Matrix')
Query: 3 0.6411464810371399
<matplotlib.image.AxesImage object at 0x7f5e022a2250>
<matplotlib.colorbar.Colorbar object at 0x7f5e022a24f0>
Text(0.5, 1.0, 'Query 4 Comparison Matrix')
"""