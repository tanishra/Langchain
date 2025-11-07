from langchain_openai import ChatOpenAI
from typing import TypedDict, Optional, Literal, Annotated
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

# Simple Schema
# class Review(TypedDict):
#     summary : str
#     sentiment : str

# Annotated Schema
class Review(TypedDict):
    key_themes : Annotated[list[str], "Write down all the key themes discussed in the review in a list"]
    pros : Annotated[Optional[list[str]],"Write down all the pros inside a list"]
    cons : Annotated[Optional[list[str]],"Write down all they cons inside a list"]
    summary : Annotated[str,"A brief summary of the review"]
    # sentiment : Annotated[str,"Return the sentiment of the reiview either positive, negative or neutral"]
    sentiment : Annotated[Literal["pos","neg"],"Return the sentiment of the review"]

structured_model = model.with_structured_output(Review)

review = """
I recently purchased the Samsung Galaxy S24 Ultra during the Amazon Prime Day sales, and I can confidently say it's the best smartphone I've ever owned. From the moment I unboxed it, I was completely captivated, and it continues to impress me daily.
As a long-time Samsung fan since my college days, finally being able to afford my dream mobile and experiencing a Samsung flagship has been a dream come true. Getting this mobile at an effective price of ₹71,250 was truly the best ₹71k I have ever invested. I am absolutely fully satisfied with this purchase.
Design & Build Quality (5/5 stars):
My first contact with this phone was love at first sight. It truly is one of the most stylish and premium-looking mobiles out there today (with the possible exception of foldable phones like the Z Fold 7). The build quality is flawless, feeling incredibly solid and well-constructed. Despite being on the heavier side, the weight distribution is exceptionally well done, making it comfortable to hold and use for extended periods.
Camera Quality (5/5 stars):
This phone boasts the most versatile camera setup I've ever encountered. Whether it's wide-angle shots, zoom, or low-light, the results are consistently stunning. A particular highlight for me is the portrait mode – it captures subjects beautifully with natural-looking bokeh, making my photos look professional.
Revolutionary AI Features (5/5 stars):
The Galaxy AI features are truly mind-blowing. Tools like Circle to Search have completely changed how I interact with information, allowing for instant lookups directly from my screen. The AI Eraser is fantastic for cleaning up photos with unwanted elements. I've only scratched the surface of what Samsung AI can do, but it's clear that once you start using these features, there's no going back. They genuinely enhance productivity and convenience.
One UI 7 - A User Experience Dream (5/5 stars):
Samsung has outdone itself with One UI 7. It is, without a doubt, the smoothest and most intuitive user interface I've experienced on any smartphone, even surpassing iOS in my opinion. The RAM management is next-level, ensuring everything runs fluidly without hitches. Plus, the customization options are incredibly extensive, allowing me to truly personalize my device.
The Indispensable S Pen (5/5 stars):
The S Pen on the S24 Ultra is a game-changer. Its Bluetooth connectivity adds so much functionality beyond just note-taking. It's incredibly precise and makes navigating, editing, and creating content a breeze. This feature alone makes the S24 Ultra a superior choice, especially when considering future models.
Samsung Wallet - My New Best Friend (5/5 stars):
I used to carry a bulky physical wallet overflowing with cards. Thanks to Samsung Wallet, my physical wallet is now virtually empty! It securely stores all my credit cards, debit cards, Aadhaar, PAN, driver's license, and more. It's incredibly easy to set up and use, and the peace of mind knowing all my essentials are safely digitized is invaluable.
Performance & Smoothness (5/5 stars):
This is truly the smoothest mobile in every aspect. The fingerprint sensor and all other functionalities work flawlessly, contributing to an incredibly fluid and responsive user experience. It was the smartphone of the year for 2024, and I firmly believe it will stand out as the best phone even in 2025, even with the emergence of new chipsets like Snapdragon 8 Elite.
Battery Life (4/5 stars):
The battery life is also quite decent, consistently providing around 7 hours of screen-on time with my usage, which gets me comfortably through a full day.
Minor Con (Charger):
The only minor drawback, which is common with many flagship phones now, is that the charger is not included in the box. My travel adapter hasn't arrived yet, so the charging speed is currently not optimal. However, this is easily remedied with the correct fast charger.
Overall Impression:
The Samsung Galaxy S24 Ultra is more than just a phone; it's a complete ecosystem that enhances every aspect of my digital life. The seamless integration of hardware, software, and AI makes for an unparalleled user experience. If you're looking for the best premium smartphone on the market, look no further. Highly, highly recommended!
"""

result = structured_model.invoke(review)

print(result)