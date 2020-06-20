import re, string
from keras.preprocessing import text
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
import collections

#Load stopwords
nltk.download('stopwords')
stop = set(stopwords.words('english'))
punctuation = list(string.punctuation)
stop.update(punctuation)


class cleaned_data:
	"""
	:param data: of type pandas.core.frame.DataFrame with 'text' in df.columns
	:attribute self.data: of type pandas.core.frame.DataFrame, including 'text' in self.data.columns
	"""
	def __init__(self, data):
		self.clean(data)


	def clean(self, data):
		#Apply denoise function on text column
		data['text']=data['text'].apply(self.denoise_text)

		words_counter = collections.Counter([word for text in data['text'] for word in text.split()])
		print('{} words.'.format(sum(words_counter.values())))
		print('{} unique words.'.format(len(words_counter)))
		print('50 most common words:')
		print('"' + '" "'.join(list(zip(*words_counter.most_common(50)))[0]) + '"')
		print()

		self.data=data


	#Removing html
	@staticmethod
	def strip_html(text):
		soup = BeautifulSoup(text, "html.parser")
		return soup.get_text()


	#Removing the square brackets
	@staticmethod
	def remove_between_square_brackets(text):
		return re.sub('\[[^]]*\]', '', text)


	# Removing URL's
	@staticmethod
	def remove_urls(text):
		return re.sub(r'http\S+', '', text)


	# Removing other characters
	@staticmethod
	def remove_other(text):
		return re.sub(r'[\\~#%&*{}/:<>?|\'"-]', '', text)


	#Removing the stopwords from text
	@staticmethod
	def remove_stopwords(text):
		final_text = []
		for i in text.split():
			if i.strip().lower() not in stop:
				final_text.append(i.strip())
		return " ".join(final_text)


	#Removing the noisy text
	def denoise_text(self,text):
		text = self.strip_html(text)
		text = self.remove_between_square_brackets(text)
		text = self.remove_urls(text)
		text = self.remove_other(text)
		#text = self.remove_stopwords(text) #not removing these since they are likely to be informative
		return text


	#Clean new text observations using same procedures used for cleaning modeling data
	def clean_new_observations(self, data):
		"""
		Apply denoise function on text column
		:param data: of type pandas.core.frame.DataFrame with 'text' in df.columns
		:return: of type pandas.core.frame.DataFrame, including 'text' in self.data.columns
		"""
		data['text']=data['text'].apply(self.denoise_text)
		return data

