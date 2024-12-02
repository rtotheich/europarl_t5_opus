{
  "cells": [
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "cCvcMTcEuqS9"
      },
      "outputs": [],
      "source": [
        "!pip install datasets"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "-Agac4_th638"
      },
      "outputs": [],
      "source": [
        "from datasets import load_dataset\n",
        "\n",
        "dataset = load_dataset(\"europarl_bilingual\", \"en-fr\")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "3ybXW0Rfi13A"
      },
      "outputs": [],
      "source": [
        "dataset"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "gYW9FyhukeY9"
      },
      "outputs": [],
      "source": [
        "dataset['train']['translation'][:5]"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "ryINRzjxq88Z"
      },
      "outputs": [],
      "source": [
        "import nltk\n",
        "from nltk.tokenize import word_tokenize"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "gJzVbyHZroQP"
      },
      "outputs": [],
      "source": [
        "nltk.download('punkt_tab')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "K2WJcgQkuNUm"
      },
      "outputs": [],
      "source": [
        "import seaborn as sns\n",
        "import matplotlib.pyplot as plt"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "vB6AZzo70LoS"
      },
      "outputs": [],
      "source": [
        "from tqdm import tqdm"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "KIDgMdw4lp_D"
      },
      "outputs": [],
      "source": [
        "overall_english_words_count_list = []\n",
        "overall_french_words_count_list = []\n",
        "\n",
        "for example in tqdm(dataset['train']['translation']):\n",
        "\n",
        "  english_sentence = example['en']\n",
        "  french_sentence = example['fr']\n",
        "\n",
        "  current_english_words_count = len(word_tokenize(english_sentence))\n",
        "  current_french_words_count = len(word_tokenize(french_sentence))\n",
        "\n",
        "  overall_english_words_count_list.append(current_english_words_count)\n",
        "  overall_french_words_count_list.append(current_french_words_count)\n",
        "\n",
        "\n",
        "sns.scatterplot(x=overall_english_words_count_list, y=overall_french_words_count_list)\n",
        "\n",
        "plt.plot([min(overall_english_words_count_list), max(overall_english_words_count_list)], [min(overall_english_words_count_list), max(overall_english_words_count_list)], color='blue', linestyle='--')\n",
        "\n",
        "plt.title(\"English vs French Sentence Lengths\")\n",
        "\n",
        "plt.xlabel('English Sentence Length')\n",
        "plt.ylabel('French Sentence Length')\n",
        "\n",
        "plt.savefig(\"english_french_sentence_lengths.png\", dpi=300, bbox_inches='tight')\n",
        "plt.show()\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "E8nbiVu3pMhg"
      },
      "outputs": [],
      "source": [
        "from google.colab import files\n",
        "files.download(\"english_italian_sentence_lengths.png\")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "5OdiO75u1BOZ"
      },
      "outputs": [],
      "source": [
        "first_example = dataset['train']['translation'][0]\n",
        "print(first_example)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "itkSaU6oAXzw"
      },
      "outputs": [],
      "source": [
        "dataset = load_dataset(\"europarl_bilingual\", \"en-it\")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "XrB4VGtJES-Q"
      },
      "outputs": [],
      "source": [
        "dataset"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "VZzM5d1rEuJ1"
      },
      "outputs": [],
      "source": [
        "dataset['train']['translation'][:5]"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "wLaAzmvcEw67"
      },
      "outputs": [],
      "source": [
        "overall_english_words_count_list = []\n",
        "overall_italian_words_count_list = []\n",
        "\n",
        "for example in tqdm(dataset['train']['translation']):\n",
        "\n",
        "  english_sentence = example['en']\n",
        "  italian_sentence = example['it']\n",
        "\n",
        "  current_english_words_count = len(word_tokenize(english_sentence))\n",
        "  current_italian_words_count = len(word_tokenize(italian_sentence))\n",
        "\n",
        "  overall_english_words_count_list.append(current_english_words_count)\n",
        "  overall_italian_words_count_list.append(current_italian_words_count)\n",
        "\n",
        "\n",
        "sns.scatterplot(x=overall_english_words_count_list, y=overall_italian_words_count_list)\n",
        "\n",
        "plt.plot([min(overall_english_words_count_list), max(overall_english_words_count_list)], [min(overall_english_words_count_list), max(overall_english_words_count_list)], color='blue', linestyle='--')\n",
        "\n",
        "plt.title(\"English vs Italian Sentence Lengths\")\n",
        "\n",
        "plt.xlabel('English Sentence Length')\n",
        "plt.ylabel('Italian Sentence Length')\n",
        "\n",
        "plt.savefig(\"english_italian_sentence_lengths.png\", dpi=300, bbox_inches='tight')\n",
        "plt.show()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "7HGZ7afrFgQ0"
      },
      "outputs": [],
      "source": [
        "files.download(\"english_italian_sentence_lengths.png\")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "cbEvEp3bvRke"
      },
      "outputs": [],
      "source": [
        "dataset = load_dataset(\"europarl_bilingual\", \"en-fr\")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "mIW9l9V3wJC2"
      },
      "outputs": [],
      "source": [
        "import nltk\n",
        "from nltk.tokenize import word_tokenize"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "DMkC5TvFwTW6"
      },
      "outputs": [],
      "source": [
        "nltk.download('punkt_tab')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "ko4iZ1_m2Ctt"
      },
      "outputs": [],
      "source": [
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "o2Jy5sE_9BMx"
      },
      "outputs": [],
      "source": [
        "from nltk.corpus import stopwords\n",
        "nltk.download('stopwords')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "Kkr0d1ALCcZ3"
      },
      "outputs": [],
      "source": [
        "from tqdm import tqdm"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "aUQ_yrvhGaGG"
      },
      "outputs": [],
      "source": [
        "stop_words = set(stopwords.words('english'))\n",
        "\n",
        "english_token_to_frequency_map = dict()\n",
        "french_token_to_frequency_map = dict()\n",
        "\n",
        "for example in tqdm(dataset['train']['translation']):\n",
        "  english_sentence = example['en']\n",
        "  french_sentence = example['fr']\n",
        "\n",
        "  english_words = word_tokenize(english_sentence)\n",
        "  french_words = word_tokenize(french_sentence)\n",
        "\n",
        "  english_words = [english_word.lower() for english_word in english_words if english_word not in stop_words and english_word.isalnum()]\n",
        "  french_words = [french_word.lower() for french_word in french_words if french_word not in stop_words and english_word.isalnum()]\n",
        "\n",
        "  for english_word in english_words:\n",
        "    english_token_to_frequency_map[english_word] = english_token_to_frequency_map.get(english_word, 0) + 1\n",
        "\n",
        "  for french_word in french_words:\n",
        "    french_token_to_frequency_map[french_word] = french_token_to_frequency_map.get(french_word, 0) + 1\n",
        "\n",
        "sorted_english_tokens = sorted(english_token_to_frequency_map.items(), key=lambda x: x[1], reverse=True)\n",
        "sorted_french_tokens = sorted(french_token_to_frequency_map.items(), key=lambda x: x[1], reverse=True)\n",
        "\n",
        "top_english_tokens = sorted_english_tokens[:10]\n",
        "top_french_tokens = sorted_french_tokens[:10]\n",
        "\n",
        "x1 = [token[0] for token in top_english_tokens]\n",
        "y1 = [token[1] for token in top_english_tokens]\n",
        "\n",
        "plt.bar(x1, y1, color='blue')\n",
        "\n",
        "plt.grid(axis='x', linestyle='--', alpha=0.7)\n",
        "\n",
        "plt.xticks(rotation=45, ha='right')\n",
        "\n",
        "plt.xlabel('English Tokens')\n",
        "plt.ylabel('Frequency')\n",
        "plt.title('Most Frequent English Tokens')\n",
        "\n",
        "plt.savefig(\"most_frequent_english_tokens_distribution.png\", dpi=300, bbox_inches=\"tight\")\n",
        "plt.show()\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "lp52ry3s2BtJ"
      },
      "outputs": [],
      "source": [
        "x2 = [token[0] for token in top_french_tokens]\n",
        "y2 = [token[1] for token in top_french_tokens]\n",
        "\n",
        "plt.bar(x2, y2, color='orange')\n",
        "\n",
        "plt.grid(axis='x', linestyle='--', alpha=0.7)\n",
        "\n",
        "plt.xticks(rotation=45, ha='right')\n",
        "\n",
        "plt.xlabel('French Tokens')\n",
        "plt.ylabel('Frequency')\n",
        "plt.title('Most Frequent French Tokens')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "odh4CNnECDdO"
      },
      "outputs": [],
      "source": [
        "stop_words = set(stopwords.words('italian'))\n",
        "\n",
        "italian_token_to_frequency_map = dict()\n",
        "\n",
        "for example in tqdm(dataset['train']['translation']):\n",
        "  italian_sentence = example['it']\n",
        "\n",
        "  italian_words = word_tokenize(italian_sentence)\n",
        "\n",
        "  italian_words = [italian_word.lower() for italian_word in italian_words if italian_word not in stop_words and italian_word.isalnum()]\n",
        "\n",
        "  for italian_word in italian_words:\n",
        "    italian_token_to_frequency_map[italian_word] = italian_token_to_frequency_map.get(italian_word, 0) + 1\n",
        "\n",
        "\n",
        "sorted_italian_tokens = sorted(italian_token_to_frequency_map.items(), key=lambda x: x[1], reverse=True)\n",
        "\n",
        "top_italian_tokens = sorted_italian_tokens[:10]\n",
        "\n",
        "x3 = [token[0] for token in top_italian_tokens]\n",
        "y3 = [token[1] for token in top_italian_tokens]\n",
        "\n",
        "plt.bar(x3, y3, color='green')\n",
        "\n",
        "plt.grid(axis='x', linestyle='--', alpha=0.7)\n",
        "\n",
        "plt.xticks(rotation=45, ha='right')\n",
        "\n",
        "plt.xlabel('Italian Tokens')\n",
        "plt.ylabel('Frequency')\n",
        "plt.title('Most Frequent Italian Tokens')\n",
        "\n",
        "plt.savefig(\"most_frequent_italian_tokens_distribution.png\", dpi=300, bbox_inches=\"tight\")\n",
        "plt.show()"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "vocabulary_sizes = {\n",
        "    'English': len(english_token_to_frequency_map),\n",
        "    'French': len(french_token_to_frequency_map),\n",
        "    'Italian': len(italian_token_to_frequency_map)\n",
        "}\n",
        "\n",
        "plt.bar(vocabulary_sizes.keys(), vocabulary_sizes.values(), color=['blue', 'orange', 'green'])\n",
        "\n",
        "plt.grid(axis='x', linestyle='--', alpha=0.7)\n",
        "\n",
        "plt.xlabel('Languages')\n",
        "plt.ylabel('Vocabulary Size')\n",
        "plt.title('Vocabulary Sizes of Languages')\n",
        "\n",
        "plt.savefig(\"vocabulary_sizes.png\", dpi=300, bbox_inches=\"tight\")\n",
        "plt.show()"
      ],
      "metadata": {
        "id": "dbcYW22UqkUe"
      },
      "execution_count": null,
      "outputs": []
    }
  ],
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}