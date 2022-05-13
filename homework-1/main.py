import math
class LanguageModel:
    unigram_map = {}
    bigram_map = {}
    total_tokens = 0
    total_sentences = 0 
    total_types = 0

    total_tokens_test_data = 0
    percent_non_occouring_tokens_q3 = 0
    percent_non_occouring_types_q3 = 0
    percent_non_occouring_tokens_q4 = 0
    percent_non_occouring_types_q4 = 0


    def __init__(self):
  
        #create unigram model for training corpus 
        self.create_unigram_model()
        self.read_training()
       
        print("Q1")
        self.total_types =  int(len(self.unigram_map.keys()) - 1 - self.unigram_map["<unk>"])
        print("Total unique word types in training corpus: ",    self.total_types)

        #Q2 total training word tokens

        print("Q2")
        self.total_tokens = sum(map(int, self.unigram_map.values()))
        self.total_tokens -= self.unigram_map["<unk>"]
        print("total tokens in training corpus: ",self.total_tokens )


        print("Q3")
        self.unseen_tokens()
        self.unseen_types()
        print(self.total_tokens_test_data)
        print("percent of unseen tokens from test data : " + str(self.percent_non_occouring_tokens_q3)+"%")
        print("percent of unseen types from test data : " + str(self.percent_non_occouring_types_q3)+"%")

        print("Q4")
        self.replace_singletons()
        self.create_bigram_model()
        self.unigram_map["<s>"] = self.total_sentences
        self.unseen_tokens_bigram()
        self.unseen_types_bigram()
        print("percent of unseen bigram tokens from test data : " + str(self.percent_non_occouring_tokens_q4)+"%")
        print("percent of unseen bigram types from test data : " + str(self.percent_non_occouring_types_q4)+"%")

        print("Q5")
        sentence = "<s> i look forward to hearing your reply . </s>"
        q5_unigram = self.logProb("unigram_mle", sentence, True )
        print("Unigram log prob for sentence: " + str(q5_unigram) + "\n")

        q5_bigram = self.logProb("bigram_mle", sentence , True )
        print("bigram log prob for sentence: " + str(q5_bigram) + "\n")

        q5_bigram_smoothed = self.logProb("bigram_smoothed", sentence,printy=True )
        print("bigram_smoothed log prob for sentence: " + str(q5_bigram_smoothed) + "\n" )

        print("Q6")
        q6_unigram = self.sentence_perplexity("unigram_mle", sentence )
        q6_bigram = self.sentence_perplexity("bigram_mle", sentence )
        q6_bigram_smoothed = self.sentence_perplexity("bigram_smoothed", sentence )
        print('unigram perplexity ',str(q6_unigram))
        print('bigram perplexity', str(q6_bigram))
        print('bigram smoothed', str(q6_bigram_smoothed))

       
        print("Q7")
        q7_unigram = self.corpus_perplexity("unigram_mle")
        q7_bigram = self.corpus_perplexity("bigram_mle")
        q7_bigram_smoothed = self.corpus_perplexity("bigram_smoothed")
        print('corpus unigram perplexity',str(q7_unigram))
        print('corpus bigram perplexity', str(q7_bigram))
        print('corpus bigram smoothed',str(q7_bigram_smoothed))

        
    def create_unigram_model(self):

        with open('train-Spring2022.txt','rt', encoding="utf8") as file:         
            for line in file:
                line_list = line.lower().split()
                for i in range(len(line_list)):
                    token = line_list[i]
                    self.unigram_map[token] = self.unigram_map.get(token, 0) + 1
                    
    def read_training(self):
        with open('train-Spring2022.txt','rt', encoding='utf_8') as training_data:
            with open('train-p.txt','wt',encoding='utf_8') as train_p:

                self.unigram_map["<s>"] = 0 
                self.unigram_map["</s>"] = 0 
                self.unigram_map["<unk>"] = 0    
            
                for line in training_data:
                    line = line.split()
                    self.total_sentences +=1
                    self.total_tokens+= len(line) + 1

                    train_p.write("<s>")
                
                    for token in line:
                        token = token.lower()
                        if self.unigram_map[token] == 1:
                            train_p.write(" <unk>")
                            self.unigram_map["<unk>"] += 1
                            
                            # del unigram_map[token]        # gonna do this later
                        else:
                            train_p.write(" " + token)
                        
                    self.unigram_map["</s>"] += 1
                    
                    train_p.write(" </s>\n")

    def unseen_tokens(self):
        with open('test.txt','rt', encoding='utf_8') as test_data:
            with open('test-p.txt','wt',encoding='utf_8') as test_p:

                unseen_tokens = 0
                total_tokens_test_data =0
                test_tokens = {}
                test_tokens['<s>'] =0
                test_tokens['<unk>']=0
                test_tokens['</s>'] =0
                for line in test_data:
                    line = line.split()
                    test_p.write("<s>")
                    test_tokens["<s>"]+=1
                        
                    for token in line:
                        token = token.lower()
                        if token not in self.unigram_map:  
                            test_p.write(" <unk>")
                            test_tokens["<unk>"]+=1
                            unseen_tokens+=1
                            
                        else:
                            test_p.write(" " + token)
                           
                        total_tokens_test_data+=1
                                      
                    test_p.write(" </s>\n")
                    test_tokens["</s>"]+=1 
            
        self.total_tokens_test_data = total_tokens_test_data
        print(total_tokens_test_data)
        self.percent_non_occouring_tokens_q3 = 100 * unseen_tokens / total_tokens_test_data

    def unseen_types(self):

        non_occouring_types = 0
        total_types_test_data = 0

        with open('test.txt','rt', encoding='utf_8') as test_data:
                types_seen= {}

                for line in test_data:
                    line = line.split()
                        
                    for token in line:
                        token = token.lower()

                        if token not in types_seen:
                            types_seen[token] = True

                            if token not in self.unigram_map:    
                                non_occouring_types+=1
                            
                            total_types_test_data+=1

            
        self.percent_non_occouring_types_q3 = 100 * non_occouring_types / total_types_test_data

    def replace_singletons(self):
        singletons = []

        for key in self.unigram_map.keys():
            if self.unigram_map[key] == 1:
                singletons.append(key)
        
        for key in singletons:
            del self.unigram_map[key]       

    def create_bigram_model(self):
        with open("train-p.txt","rt",encoding="utf8") as train:
            for line in train:
                line_list = line.lower().split()
                for i in range(1, len(line_list)):
                    token = line_list[i-1] + " " + line_list[i]
                    self.bigram_map[token] = self.bigram_map.get(token, 0) + 1

    def unseen_tokens_bigram(self):
        with open('test-p.txt','rt', encoding='utf_8') as test_data:
                test_tokens = {}
                unseen_tokens_bigram = 0
                for line in test_data:
                    line_list = line.split()
                    for i in range(1, len(line_list)):
                        token = line_list[i-1] + " " + line_list[i]
                        if token not in self.bigram_map:  
                            unseen_tokens_bigram+=1
                            test_tokens[token] = True

        
        self.percent_non_occouring_tokens_q4 = 100 * unseen_tokens_bigram / self.total_tokens_test_data

    def unseen_types_bigram(self):

        non_occouring_types = 0
        total_types_test_data = 0
        types_seen = {}

        with open('test-p.txt','rt', encoding='utf_8') as test_data:

                for line in test_data:
                    line_list = line.split()
                    for i in range(1, len(line_list)):
                        token = line_list[i-1] + " " + line_list[i]

                        if token not in types_seen:
                            types_seen[token] = True

                            if token not in self.bigram_map:    
                                non_occouring_types+=1
                            
                            total_types_test_data+=1

            
        self.percent_non_occouring_types_q4 = 100 * non_occouring_types / total_types_test_data

    def logProb(self, type_of_model: str, line:str, printy:bool):
        total_log = 0
        tokens = line.split()
        for i in range(1,len(tokens)):
            if type_of_model== "unigram_mle":
                t = tokens[i]
                try:
                    prob_of_token = self.unigram_map[t] / self.total_tokens
               
                except :
                    prob_of_token = self.unigram_map["<unk>"] / self.total_tokens

            elif type_of_model == "bigram_mle":
                t = (tokens[i-1] + " " + tokens[i])
                try:
                    prob_of_token = self.bigram_map[t]/self.unigram_map[tokens[i-1]]
                except :
                    logProb = 'undefined'
                    if printy:
                        print("log2(P( " + t + " ) = " + str(logProb) ) 
                    return "undefined"
                
            else:
                t = (tokens[i-1] + " " + tokens[i])
                try:
                    prob_of_token = (self.bigram_map[t] + 1) / ( self.unigram_map[tokens[i-1]] + self.total_types) 
                except :
                    try:
                        prob_of_token = 1 / ( self.unigram_map[tokens[i-1]] + self.total_types) 
                    except :
                        prob_of_token = 1 / (self.unigram_map["<unk>"] + self.total_types )

            logProb = math.log2(prob_of_token)
            
            
            total_log += logProb
            if printy:
                print("log2(P( " + t + " ) = " + str(logProb) ) 
        

        return total_log               

    def sentence_perplexity(self,type_of_model: str, line: str):

        
        tokens = line.split()
        numTokens = int(len(tokens) - 1)
      
    
        for i in range(1,len(tokens)):
            if type_of_model== "unigram_mle":
                    
                    averaged_log = self.logProb('unigram_mle', line, False)/numTokens
                    

            elif type_of_model == "bigram_mle":
                
                try:
                    averaged_log = self.logProb('bigram_mle', line, False)/numTokens                
                   
                except:
                    perplexity = "undefined"
                    return "undefined"
                    
            else:
                
                try:
                    averaged_log = self.logProb('bigram_smoothed', line, False)/numTokens

                except:
                    averaged_log = 1 / ( self.unigram_map[tokens[i-1]] + self.total_types) 

 
            
        perplexity = pow(2,-1 * averaged_log)
        return perplexity               



    def corpus_perplexity(self, type_of_model: str):

        with open('test-p.txt','rt',encoding='utf-8') as file:
            sum = 0
            for line in file:
                
    
                if type_of_model== "unigram_mle":
                        sum+= self.logProb('unigram_mle', line, False)
                        

                elif type_of_model == "bigram_mle":
                    try:
                        sum+= self.logProb('bigram_mle', line, False)                   
                        
                    except:
                        perplexity = "undefined"
                        return "undefined"
                        
                else:
                        sum+= self.logProb('bigram_smoothed', line, False)

                    

        averaged_log = sum/(self.total_tokens_test_data + 100)
        perplexity = pow(2,-1 * averaged_log) 
        return perplexity    
            

if __name__ == "__main__":
      l = LanguageModel()
      


