from sklearn.datasets import fetch_20newsgroups
import pandas as pd


categories = {'alt.atheism': 'religion',\
	'comp.graphics': 'computers',\
	'comp.os.ms-windows.misc': 'computers',\
	'comp.sys.ibm.pc.hardware': 'computers',\
	'comp.sys.mac.hardware': 'computers',\
	'comp.windows.x': 'computers',\
	'rec.autos': 'recreation',\
	'rec.motorcycles': 'recreation',\
	'rec.sport.baseball': 'recreation',\
	'rec.sport.hockey': 'recreation',\
	'sci.crypt': 'science',\
	'sci.electronics': 'science',\
	'sci.med': 'science',\
	'sci.space': 'science',\
	'soc.religion.christian': 'religion',\
	'talk.politics.guns': 'politics',\
	'talk.politics.mideast': 'politics',\
	'talk.politics.misc': 'politics',\
	'talk.religion.misc': 'religion'\
	} #does not include 'misc.forsale'


def load_data():
	"""
	:return: df, of type pandas.core.frame.DataFrame, with df.columns: Index(['text', 'subject'], dtype='object')
	"""
	bunch = fetch_20newsgroups(subset='all', categories=list(categories.keys()), remove=('headers', 'footers', 'quotes'))
	df = pd.DataFrame({'text':bunch.data, 'subject':bunch.target})
	df['subject'] = df['subject'].map(lambda i: get_target_name(i, bunch.target_names))
	df['subject'] = df['subject'].map(categories)
	pd.set_option('display.max_colwidth', 20)
	print('Number of rows:')
	print(len(df))
	print()
	print('Subject types:')
	print(df.subject.value_counts())
	print('Top of dataset:')
	print(df.head())
	print()
	print('Check for nan values:')
	print(df.isna().sum())
	print()
	return(df)


def get_target_name(index, classes):
	return classes[index]
