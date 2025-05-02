from flask import Flask, request, jsonify
from urllib.parse import urlparse, parse_qs
from utilities_fucns import comment_scrap, sentiment_analysis,generate_graphs, insight, generate_dashboard_html
from flask import render_template
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# driver_path = r"D:\Sharda\IIT BHU\chromedriver\chromedriver.exe" # Replace with the actual path to chromedriver
AI_INSIGHTS = ''
def sentiment_analysis_pipiline(video_id):
    percentages = sentiment_analysis(f"{video_id}.csv")
    generate_graphs(f"{video_id}.csv")

    return percentages

@app.route('/receive_url', methods=['POST','GET'])
def receive_url():
    global AI_INSIGHTS
    data = request.json
    youtube_url = data.get('youtube_url')
    parsed_url = urlparse(youtube_url)
    if not youtube_url:
        return jsonify({"error": "No URL provided"}), 400
    
    video_id = parse_qs(parsed_url.query).get("v", [None])[0]
    print(video_id)
    print("Call scrap function")
    numberOfComments = comment_scrap(video_id,"scraped_comments/")
    print("End scrap function")
    if type(numberOfComments)==int:
        print(f"Total {numberOfComments} Comments Exacted.")
    else:
        return jsonify({'message':"Comments extaction failed."})
    print(f"Total {numberOfComments} Comments Exacted.")

    print("Sentiment Analysis Initiate.")
    percentages = sentiment_analysis_pipiline(video_id)
    print('Sentiment Analysis Done')

    print("Extracting Insights...")
    AI_INSIGHTS = insight(video_id)
    print("Extracted Insights...")
    generate_dashboard_html(video_id,AI_INSIGHTS)

    return jsonify({
            "message": "Comments extracted and analyzed successfully!",
            "comments_count": numberOfComments,
            "sentiments": percentages,  
            'video_id':video_id,
            'suggestions':AI_INSIGHTS
        })




if __name__ == '__main__':
    app.run()
