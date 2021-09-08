from TextClassify import TextDeal

if __name__ == '__main__':
    test = TextDeal()
    text_predict = test.classify_text("PLATINUM has renamed rar.exe to avoid detection.")
    print(text_predict["total_techniques"])