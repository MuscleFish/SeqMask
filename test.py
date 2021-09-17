from TextClassify import TextDeal, TextImportance

if __name__ == '__main__':
    # test for classify tactics and techniques
    test = TextDeal()
    text_predict = test.classify_text("PLATINUM has renamed rar.exe to avoid detection.")
    print(text_predict["total_techniques"])

    # test for pick uo keywords
    # test = TextImportance()
    # text_scores = test.score_text("PLATINUM has renamed rar.exe to avoid detection.")
    # print(text_scores)