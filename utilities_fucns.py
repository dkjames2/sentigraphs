import os
from dotenv import load_dotenv
from googleapiclient.discovery import build as GoogleAPIClientBuild
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
from nltk.corpus import stopwords
import google.generativeai as genai

load_dotenv()
api_service_name = "youtube"
api_version = "v3"
API_KEY = os.getenv("YOUTUBE_API_KEY")



def comment_scrap(video_id,dir_path=''):

    file_path = dir_path+f"{video_id}.csv"
    print("Start Scrapping")
    try:
        youtube = GoogleAPIClientBuild(
            api_service_name,api_version,developerKey=API_KEY
        )

        
        comments = []
        next_page_token = None

        while True:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId = video_id,
                maxResults = 100,
                pageToken = next_page_token
                )
            
            response = request.execute()


            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']
                comments.append([
                    comment['likeCount'],
                    comment['textDisplay']
                ])

            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break

        df = pd.DataFrame(comments,columns=['like_count','comment'])
        df.to_csv(file_path,index=False)
        print("Comments Exacted. And to this file path: "+file_path)
        return len(comments)
    except Exception as e:
        print(e)
        return f"An Error has occur: {e}"
    








sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
def classify_sentiment(text):
    if not isinstance(text, str) or text.strip() == "":
        return "neutral"  

    result = sentiment_pipeline(text[:512])[0]  
    label = result["label"]
    
    # Map 5-class model labels to our desired labels
    stars = int(label[0])  # Extract the first character as an integer (1 to 5)
    
    if stars >= 4:
        return "positive"
    elif stars < 2:
        return "negative"
    else:
        return "neutral"





def preprocess_text(text, remove_stopwords=False):

    if not isinstance(text, str):
        return ""
    # Remove html tags
    text = re.sub(r'<.*?>','',text)

    # To remove urls
    text = re.sub(r'http\s+|www\s+|https\s+', '', text)

    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters, punctuation, and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def sentiment_analysis(file_name,folder_name='scraped_comments',output_folder_name='sentiment_analyzed_comments'):
    if not os.path.exists(output_folder_name):
        os.makedirs(output_folder_name)

    file_path = os.path.join(folder_name,file_name)
    df = pd.read_csv(file_path)
    print(df.head())
    df['clean_comment'] = df['comment'].astype(str).apply(preprocess_text)
    df['sentiment'] = df["clean_comment"].apply(classify_sentiment)
    
    # to save sentiment analyzed csv file.
    output_file_path = os.path.join(output_folder_name,file_name)
    df.to_csv(output_file_path)
    
    print(f"Sentiment analysis completed. Results saved in '{output_file_path}'.")
    percentages = df['sentiment'].value_counts(normalize=True) * 100
    return list(map(lambda x:round(x,1),list(percentages.to_dict().values())))



def generate_graphs(file_name,folder_name='sentiment_analyzed_comments',output_folder = 'static/graphs'):
    file_path = os.path.join(folder_name,file_name)
    output_folder_path = os.path.join(output_folder,file_name[:-4])
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    df = pd.read_csv(file_path)
    
    # Sentiment Distribution Bar Chart
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="sentiment", palette={"positive": "green", "neutral": "gray", "negative": "red"})
    plt.title("Sentiment Distribution of Comments")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.savefig(os.path.join(output_folder_path, "countPlot.png"))
    plt.close()
    
    # Pie Chart for Sentiment Proportion
    sentiment_counts = df["sentiment"].value_counts()
    plt.figure(figsize=(6, 6))
    plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=["green", "red", "gray"])
    plt.title("Sentiment Proportion")
    plt.savefig(os.path.join(output_folder_path,"pieChart.png"))
    plt.close()
    
    # Word Cloud
    text = " ".join(df["comment"].dropna())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("Most Common Words in Comments")
    plt.savefig(os.path.join(output_folder_path, "wordCount.png"))
    plt.close()
    
    # Comment Length Distribution
    df["Comment_Length"] = df["comment"].astype(str).apply(len)
    plt.figure(figsize=(8, 5))
    sns.histplot(df["Comment_Length"], bins=30, kde=True, color='blue')
    plt.title("Comment Length Distribution")
    plt.xlabel("Length of Comment")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(output_folder_path, "histplot.png"))
    plt.close()
    
    print(f"Graphs saved in {output_folder_path}")
    return output_folder_path



def insight(video_id):
    try:
        # Load CSV file
        df = pd.read_csv(f'sentiment_analyzed_comments/{video_id}.csv')
        comments = df['clean_comment'].dropna().tolist()

        combined_comments = "\n".join(comments)

        # Join all comments into one prompt string
        combined_comments = "\n".join(df["clean_comment"].dropna().astype(str).tolist())

        # Define the prompt template
        prompt = f"""
            You are an AI that analyzes social media comments and extracts insights and actionable suggestions.

            Instructions:
            - Analyze the comments below.
            - Output only two sections:
            1. Key Insights (50-80 words max): Summarize major themes, common sentiments, and recurring issues.
            2. Suggestions: Provide clear, constructive recommendations if any patterns of feedback demand it. Use bullet points.

            Rules:
            - Focus on patterns, not one-off comments.
            - Ignore spam or generic praise unless it's a trend.
            - Be concise. No need to mention you're an AI.

            Comments:
            {combined_comments}
            """

        # Initialize the Gemma 3 model
        model = genai.GenerativeModel(model_name="gemma-3-12b-it")  # Adjust model as needed

        # Generate the response
        response = model.generate_content(prompt[:5000])

        # Store the AI response in the global variable
        insights = re.sub(r"[#*]", "", str(response.text.strip()))
        print(insights)
        return insights

    except Exception as e:
        print(f"Error: {e}")

import os

def generate_dashboard_html(video_id, ai_insights,output_dir='dashboards'):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # HTML content with placeholders replaced
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Sentilytics Dashboard</title>
    <style>
        body {{
            background-color: #121212;
            color: #ffffff;
            font-family: 'Segoe UI', sans-serif;
            margin: 0;
            padding: 0;
        }}
        .container {{
            padding: 2rem;
            max-width: 1200px;
            margin: auto;
        }}
        h1 {{
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }}
        .subtitle {{
            color: #bbbbbb;
            margin-bottom: 2rem;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
        }}
        .card {{
            background-color: #1f1f1f;
            border-radius: 12px;
            padding: 1rem;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.3);
        }}
        img {{
            width: 100%;
            border-radius: 8px;
        }}
        .section-title {{
            font-size: 1.2rem;
            margin-bottom: 0.5rem;
            color: #88c0d0;
        }}
        .suggestions {{
            line-height: 1.6;
            color: #dddddd;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Sentilytics Dashboard</h1>
        <div class="subtitle">Real-Time Insights from Social Media Comments</div>
        <div class="grid">
            <div class="card">
                <div class="section-title">Sentiment Count Plot</div>
                <img src="static/graphs/{video_id}/countPlot.png" alt="Count Plot">
            </div>
            <div class="card">
                <div class="section-title">Sentiment Distribution Pie Chart</div>
                <img src="static/graphs/{video_id}/pieChart.png" alt="Pie Chart">
            </div>
            <div class="card">
                <div class="section-title">Word Count</div>
                <img src="static/graphs/{video_id}/wordCount.png" alt="Word Count">
            </div>
            <div class="card">
                <div class="section-title">Comment Length Histogram</div>
                <img src="static/graphs/{video_id}/histplot.png" alt="Histogram">
            </div>
        </div>
        <div class="grid" style="margin-top: 2rem;">
            <div class="card">
                <div class="section-title">💡 AI Insights and Suggestions</div>
                <div class="suggestions">
                <pre>{ai_insights}</pre>
                </div>
            </div>
        </div>
    </div>
</body>
</html>'''

    # Save to file
    filepath = os.path.join(output_dir, f"{video_id}.html")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Dashboard HTML generated: {filepath}")
