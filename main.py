from collections import Counter
import re
from textblob import TextBlob
from nltk import ngrams
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
from tqdm import tqdm
import time

# Function to read and process the text file
def process_text(file_path):
    print("Reading and processing text file...")
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read().lower()  # Convert to lowercase for uniformity
    words = re.findall(r'\b\w+\b', text)  # Tokenize words
    print("Text file processed. Total words found:", len(words))
    return text, words

# Function to count word frequency with progress tracking
def word_frequency(words):
    print("\nCounting word frequency...")
    word_count = Counter()

    # Progress bar using tqdm
    for word in tqdm(words, desc="Processing words", unit="word"):
        word_count[word] += 1
    print("Word frequency counting complete.")
    sorted_word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    return sorted_word_count

# Function to perform sentiment analysis
def sentiment_analysis(text):
    print("\nPerforming sentiment analysis...")
    blob = TextBlob(text)
    sentiment = blob.sentiment
    print("Sentiment analysis complete.")
    return sentiment

# Function to extract common phrases (bigrams and trigrams) with progress
def common_phrases(words, n):
    print(f"\nExtracting common {n}-grams...")
    n_grams = ngrams(words, n)
    n_grams_freq = Counter()

    # Processing with tqdm for progress tracking
    for n_gram in tqdm(n_grams, desc=f"Processing {n}-grams", unit="n-gram"):
        n_grams_freq[n_gram] += 1
    print(f"Common {n}-grams extraction complete.")
    return n_grams_freq.most_common(10)

# Function to create a word cloud
def create_word_cloud(word_count):
    print("\nGenerating word cloud...")
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(word_count))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()
    print("Word cloud generated.")

# Function to plot word frequency distribution
def plot_word_frequency(word_count, num_words=10):
    print(f"\nPlotting top {num_words} most frequent words...")
    top_words = dict(word_count[:num_words])
    sns.barplot(x=list(top_words.values()), y=list(top_words.keys()))
    plt.title(f'Top {num_words} Most Frequent Words')
    plt.xlabel('Frequency')
    plt.ylabel('Words')
    plt.show()
    print("Word frequency plot generated.")

# Main analysis function
def analyze_text(file_path):
    print("Starting text analysis...")

    # Simulate loading for user feedback (optional)
    time.sleep(1)

    # Step 1: Read and process the text
    text, words = process_text(file_path)

    # Step 2: Word frequency analysis
    sorted_word_count = word_frequency(words)
    most_repeated_words = sorted_word_count[:10]
    least_repeated_words = sorted_word_count[-10:]

    # Step 3: Sentiment analysis
    sentiment = sentiment_analysis(text)

    # Step 4: Common phrases (bigrams and trigrams)
    bigrams = common_phrases(words, 2)
    trigrams = common_phrases(words, 3)

    # Step 5: Visualization (word cloud and frequency distribution)
    create_word_cloud(sorted_word_count)
    plot_word_frequency(sorted_word_count)

    # Step 6: Output the results
    print("\nAnalysis complete. Saving results to output files...")

    # Save word frequency count to a separate file
    with open('word_frequencies.txt', 'w') as word_freq_file:
        word_freq_file.write("Word Frequency Count:\n")
        for word, freq in sorted_word_count:
            word_freq_file.write(f"{word}: {freq}\n")
    print("Word frequencies saved to 'word_frequencies.txt'.")

    # Save other results to a text file
    with open('analysis_output.txt', 'w') as output_file:
        output_file.write("Most Repeated Words:\n")
        for word, freq in most_repeated_words:
            output_file.write(f"{word}: {freq}\n")
        output_file.write("\nLeast Repeated Words:\n")
        for word, freq in least_repeated_words:
            output_file.write(f"{word}: {freq}\n")
        output_file.write(f"\nSentiment Polarity: {sentiment.polarity}, Subjectivity: {sentiment.subjectivity}\n")
        output_file.write("\nCommon Bigrams:\n")
        for phrase, freq in bigrams:
            output_file.write(f"{' '.join(phrase)}: {freq}\n")
        output_file.write("\nCommon Trigrams:\n")
        for phrase, freq in trigrams:
            output_file.write(f"{' '.join(phrase)}: {freq}\n")
    print("Results saved to 'analysis_output.txt'.\n")

    print("Text analysis completed successfully.")

# Run the analysis on your text file
file_path = 'supertf speeches and links.txt'  # Replace with the path to your text file
analyze_text(file_path)
