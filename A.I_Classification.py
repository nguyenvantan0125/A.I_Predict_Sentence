    # import module 
import module_AI as ai

#%%
    # read file data set - đọc file data set csv
df = ai.readcsv("data_train.csv")
df.rename(columns={0:'feature',1:'label'},inplace=True)
    # create data set - tạo data set
corpus = df['feature'].values.tolist()
y = df['label'].values.tolist()
    # token Vietnamese - tách từ tiếng việt
corpus = ai.vi_tokenizer(corpus)
    # create pipeline - tạo pipeline
    # .pipeline_svc , .pipeline_multinomialNB 
pl = ai.pipeline_multinomialNB()
    # fit = pipeline
pl.fit(corpus,y)


#%%
    # run AI with pipeline 
run = ai.interface(pl)
run.excute()


    

    
    
    
    
    
    
    
    
    