# import dependencies
import random as rn
from simpful import *
import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")
import string
import pickle

# natural language processing block
import spacy
nlp = spacy.load('en_core_web_sm')
import string
punct = string.punctuation
from spacy.lang.en.stop_words import STOP_WORDS
stopwords = list(STOP_WORDS) # list of stopwords

# creating a function for data cleaning
def text_data_cleaning(sentence):
  doc = nlp(sentence)

  tokens = [] # list of tokens
  for token in doc:
    if token.lemma_ != "-PRON-":
      temp = token.lemma_.lower().strip()
    else:
      temp = token.lower_
    tokens.append(temp)
 
  cleaned_tokens = []
  for token in tokens:
    if token not in stopwords and token not in punct:    # Stopwords and punctuation removal
      cleaned_tokens.append(token)
  return cleaned_tokens


# sentiment analysis model
import pickle
# Load the saved model using pickle
with open('C://Users/Hoashalarajh/Downloads/interAct_python_package/emotion_detector.pkl', 'rb') as file:
    loaded_model = pickle.load(file)


class hstn():
    
    # initiating the interaction counter
    interaction_counter = 0
    # initiating user engagement
    user_input = []
    currentSuperStateTransition = False
    userEngagement = 0
    userOpinion_positive = []
    nextSuperState = 2 # 0: supporting user; 1: listening to user; 2: main task
    number_of_interactions_fixed = 6

    # rules
    # first system looks for 6 interactions
    # after the a superstate breaks, switches to other superstate then, it willl looks for next three interactions before making IM decision 

    # Main state - Greeting
    def greeting(self):
        greeting_responses = ["Hi!","Good morning","Hey, what's up?","Greetings!","How are you doing?","Hi, there! How's your day going?",
                              "Hey, nice to see you!","Well, hello to you too!","Hi, how can I assist you today?","Hello, friend!",
                              "How's everything?","Hey, it's great to see you again!","Hi there, what's new?","How's your day been so far?",
                              "Hello, how's life treating you?"]
        print (greeting_responses[rn.randint(0,len(greeting_responses)-1)])

    # super state - about lecture
    def about_lecture(self, sentence):
        about_lec = sentence
        #print (about_lec)
        print (about_lec)
        return about_lec 
    
    # super state - task organization
    def task_organization(self, sentence):
        # superstate - Task organization
        task_org_responses = sentence
        return task_org_responses
    
    # superstate - Ask about student interaction
    def student_interaction(self, sentence):
        # superstate - Student Interaction
        student_int_responses = sentence
        return student_int_responses
    
    # superstate - Ask about Clarity 
    def clarity(self, sentence):
        # superstate ask about Clarity of the lecture
        clarity_responses = sentence
        return clarity_responses
    
    # listeing to user
    def listening_to_user(self, sentence):
        listening_response = sentence
        return listening_response

    # supporting to user
    def Supporting_user(self, sentence):
        supporting_user = sentence
        return supporting_user
    
    # Main state - Concluding
    def concluding(self):
        concluding_responses = ["Thank you for your insights and valuable input. Your perspective is greatly appreciated", "I appreciate your time and thoughtful responses. Thank you for sharing your thoughts with me",
                                "Thank you for taking the time to discuss these matters with me. Your input has been instrumental", "I'm grateful for the opportunity to hear your thoughts on this. Thank you for your time and thoughtful contributions",
                                "Your time and expertise are highly valued. Thank you for engaging in this discussion with me", "Thank you for spending your valuable time sharing your experiences and opinions. It has been a meaningful conversation",
                                "I want to express my gratitude for your time and the enriching conversation we've had. Thank you for your insights", "Thank you for your thoughtful responses. Your time and input have been invaluable to this discussion",
                                "I appreciate the depth of our conversation. Thank you for dedicating your time and sharing your perspective with me", "Thank you for your time and openness in discussing these matters. It has been a pleasure engaging in this conversation with you"]
        print (concluding_responses[rn.randint(0,len(concluding_responses)-1)])

    # calulating the fuzzy output value considering emotional cue and user engagement as input
    def calc_fuzzy(self, val1, val2):
        # A simple fuzzy inference system for the tipping problem
        # A simple fuzzy inference system for the tipping problem
        # Create a fuzzy system object
        FS = simpful.FuzzySystem()

        # Define fuzzy sets and linguistic variables
        S_1 = simpful.FuzzySet(points=[[0., 1.], [0.19, 1.], [0.49, 0.0]], term='Negative')
        S_2 = simpful.FuzzySet(points=[[0.203 , 0.], [0.503 , 1.], [0.8023, 0.]], term="Neutral")
        S_3 = simpful.FuzzySet(points=[[0.5011, 0.], [0.798 , 1.], [1., 1.]], term="Positive")
        FS.add_linguistic_variable("Emotion", simpful.LinguisticVariable([S_1, S_2, S_3], concept="Recent Emotion", universe_of_discourse=[0,1]))

        F_1 = simpful.FuzzySet(points=[[0., 1.], [0.097, 1.0], [0.2974, 0]], term="Very_low")
        F_2 = simpful.FuzzySet(points=[[0.157, 0.0], [0.34, 1.0], [0.5043, 0.0]], term="Low")
        F_3 = simpful.FuzzySet(points=[[0.4, 0.0], [0.5, 1], [0.6008, 0.0]], term="Medium")
        F_4 = simpful.FuzzySet(points=[[0.5, 0.0], [0.758, 1.0], [0.851, 0.0]], term="High")
        F_5 = simpful.FuzzySet(points=[[0.702, 0.0], [0.9009, 1], [1.0, 1.0]], term="Very_High")
        FS.add_linguistic_variable("Engagement", simpful.LinguisticVariable([F_1, F_2, F_3, F_4, F_5], concept="User Engagement", universe_of_discourse=[0,1]))

        # Define output crisp values
        FS.set_crisp_output_value("Supporting", 0.1)
        FS.set_crisp_output_value("Listening", 0.5)
        FS.set_crisp_output_value("Main", 1.0)

        # Define function for generous tip (food score + service score + 5%)
        #FS.set_output_function("generous", "Food+Service+5")

        # Define fuzzy rules
        R1 = "IF (Emotion IS Negative) AND (Engagement IS Very_low) THEN (Tip IS Supporting)"
        R2 = "IF (Emotion IS Negative) AND (Engagement IS Low) THEN (Tip IS Supporting)"
        R3 = "IF (Emotion IS Negative) AND (Engagement IS Medium) THEN (Tip IS Supporting)"
        R4 = "IF (Emotion IS Negative) AND (Engagement IS High) THEN (Tip IS Listening)"
        R5 = "IF (Emotion IS Negative) AND (Engagement IS Very_High) THEN (Tip IS Listening)"
        R6 = "IF (Emotion IS Neutral) AND (Engagement IS Very_low) THEN (Tip IS Supporting)"
        R7 = "IF (Emotion IS Neutral) AND (Engagement IS Low) THEN (Tip IS Supporting)"
        R8 = "IF (Emotion IS Neutral) AND (Engagement IS Medium) THEN (Tip IS Listening)"
        R9 = "IF (Emotion IS Neutral) AND (Engagement IS High) THEN (Tip IS Main)"
        R10 = "IF (Emotion IS Neutral) AND (Engagement IS Very_High) THEN (Tip IS Main)"
        R11 = "IF (Emotion IS Positive) AND (Engagement IS Very_low) THEN (Tip IS Supporting)"
        R12 = "IF (Emotion IS Positive) AND (Engagement IS Low) THEN (Tip IS Supporting)"
        R13 = "IF (Emotion IS Positive) AND (Engagement IS Medium) THEN (Tip IS Listening)"
        R14 = "IF (Emotion IS Positive) AND (Engagement IS High) THEN (Tip IS Main)"
        R15 = "IF (Emotion IS Positive) AND (Engagement IS Very_High) THEN (Tip IS Main)"
        FS.add_rules([R1, R2, R3,R4,R5,R6,R7,R8,R9,R10,R11,R12,R13,R14,R15])


        # Set antecedents values
        FS.set_variable("Emotion", val1)
        FS.set_variable("Engagement", val2)

        # Perform Sugeno inference and print output
        return(FS.Sugeno_inference(["Tip"]))
    
    # defining a function for calculating user engagement
    def calc_user_eng(self, array):
        # assuming array consists of all the user responses
        last_six_int = array[-6:]
        # total words in last 6 interactions
        total_words = 0
        # a way if getting total number of words in recent 6 interactions
        for i in last_six_int:
            words = i.split()
            total_words = total_words + len(words)
        # getting number of long responses
        long_response = 0
        for i in last_six_int:
            if len(i.split()) >= (total_words / 6):
                long_response = long_response + 1
        return (long_response / 6)
    
    # defining a function for calculating user opinon
    def calc_user_opn(self, array):
        # assuming array consist of all of the user opnions for their responses
        last_six_int = array[-6:]
        # the above have a mix of 'positive' and 'negative'
        pos_opn = 0
        for i in last_six_int:
            if i == "positive":
                pos_opn = pos_opn + 1
        return (pos_opn / 6)

    # a method to loop through all sentences in a list
    def loop_all(self, sentence, loaded_model):
        #print (type(sentence))
        if len(sentence) > 0:
            for i in sentence:
                print (i)
                user_input_instant = self.get_user_input()
                opinion = loaded_model.predict([user_input_instant])
                print (opinion)
                self.userOpinion_positive.append(opinion)

                self.interaction_counter = self.interaction_counter + 1
                if (self.interaction_counter) >= self.number_of_interactions_fixed:
                    # once the number of interaction exceeded 6 then system looks for next 3 interactions before changing interaction mode
                    #self.number_of_interactions_fixed = 6
                    
                    # getting interaction modes
                    self.state_switch(self.userOpinion_positive, self.user_input, self.currentSuperStateTransition)
                    # decison making based on interaction modes
                    if (self.nextSuperState == 2):
                        pass

                    elif (self.nextSuperState == 1):
                        self.interaction_counter = 0
                        self.loop_all(self.listening_to_user, loaded_model)

                    elif (self.nextSuperState == 0):
                        self.interaction_counter = 0
                        self.loop_all(self.Supporting_user, loaded_model)

                    elif (self.nextSuperState == 3):
                        break

                    else:
                        pass        
        else:
            # ignoring the empty list 
            pass

    # defining state switching condition
    def state_switch(self, userOpinion, userEngagement, currentSuperstateTransition):
        userOpinion = self.calc_user_opn(userOpinion)
        print (userOpinion)
        userEngagement = self.calc_user_eng(userEngagement)
        print (userEngagement)
        interactionMode = self.calc_fuzzy(userOpinion, userEngagement)
        interactionMode = interactionMode['Tip']
        if currentSuperstateTransition == False:

            if interactionMode <= 0.32:
                self.nextSuperState = 0
                self.currentSuperStateTransition = True
            elif interactionMode <= 0.74:
                self.nextSuperState = 1
                self.currentSuperStateTransition = True
            else:
                self.nextSuperState = 2
        else:
            if interactionMode <= 0.74:
                self.nextSuperState = 3 # 3: denotes to break from the current super state and switch to the next direct superstate
                self.currentSuperStateTransition = False
                self.interaction_counter = 3
                
            else:
                self.nextSuperState = 2 # 2: go ahead with the main task

    # run the overall system
    def run_system(self):
        self.greeting()
        self.get_user_input()
        self.loop_all(self.about_lecture, loaded_model)
        self.loop_all(self.task_organization, loaded_model)
        self.loop_all(self.student_interaction, loaded_model)
        self.loop_all(self.clarity, loaded_model)
        self.concluding()

    # get the user input
    def get_user_input(self):
        user_input_instant = input("Type your reseponse here: ")
        self.user_input.append(user_input_instant)
        return user_input_instant

    # getting the user responses at once
    def get_user_response(self):
        return (self.user_input)



text1 = ["What were the main key points or takeaways from today's lecture?","Can you summarize the lecture in your own words?",
                            "Were there any concepts or ideas discussed in the lecture that you found challenging or unclear? If so, please explain.",
                            "How do you think the material covered in this lecture relates to previous lectures or topics we've discussed in the course?",
                            "Can you provide an example or real-life application of one of the concepts discussed in the lecture?",
                            "Did the lecturer mention any open questions or areas of research related to the topic? What are your thoughts on those?",
                            "How would you apply what you've learned in today's lecture to solve a specific problem or scenario?",
                            "Were there any interesting or surprising insights from the lecture that you'd like to share or discuss further?",
                            "Can you identify any connections between the content of this lecture and current events or trends in the field?",
                            "Do you have any questions or concerns about the material that was covered today?"]

text2 = ["Speed of the sessions/time management were reasonable","Relevant course matter was provided",
                                "Recommended useful textbooks, websites, periodicals","Syllabus was substantially covered in the class",
                                "Number of worked examples and tutorials were adequate","The lecturer promotes self studies by the student",
                                "Practical applications relevant to the module were discussed","Professional approach of the lecturer was high",
                                "The lecturer advised regarding evaluation","Continuous assessment helps the learning process",
                                "Feedback on continuous assessment was helpful to identify my weaknesses before he final examination"]
text3 = ["To what extent did the lecturer facilitate and encourage active discussions among students?", "How effective was the lecturer in creating an environment conducive to student participation?",
                                 "How frequently did the lecturer recognize and appreciate student responses?", "How well did the lecturer manage the balance between encouraging participation and maintaining order?",
                                 "How actively did the lecturer encourage students to ask questions?", "How constructive and helpful was the feedback given to students during discussions?",
                                 "How well did the lecturer adapt their teaching based on student responses?", "What strategies did the lecturer employ to enhance student engagement during discussions?",
                                 "How effective were these strategies in promoting a dynamic and engaging learning environment?", "ow accessible and approachable did the lecturer appear during interactions with students?",
                                 "How did the lecturer balance individual and group interactions for good learning experience?"]
text4 = ["Was the pace of the delivery consistent throughout the session?", "Did the lecturer maintain an appropriate speed, allowing for easy understanding?",
                             "How effective were the black/white board or PowerPoint presentations in conveying information?", "Could you easily hear and understand the lecturer throughout the session?",
                             "How clear were the verbal explanations provided by the lecturer?", "Were complex concepts articulated in a way that was easy to comprehend?",
                             "How effectively did the lecturer address questions to ensure understanding?", "How responsive was the lecturer to feedback provided by students?",
                             "How well did the lecturer accommodate diverse learning preferences?"]
text5 = ["I agree with your opinion", "Please share more about your opinion, that will be useful for future students",
                             "feel free to talk about any inconvenience you have"]
text6 = ["Please don't miss out any lectures each and every lectures will be valuable", "At least you will learn something in the lecture rather than avoiding it",
                          "In lecutres try to concentrate on what is explained", "You can view the lecture materials before the lecture is conducted then you can understand more on lecutre itself"]



# creating an object of hstn class
dialogue_system = hstn()

# storing the text1 in about_lecture method and others for building the diaogue system
dialogue_system.about_lecture = text1
dialogue_system.clarity = text4
dialogue_system.student_interaction = text3
dialogue_system.task_organization = text2
dialogue_system.listening_to_user = text5
dialogue_system.Supporting_user = text6

# running the overall system
dialogue_system.run_system()

# saving student response in a text file
# List of strings
strings_list = dialogue_system.get_user_response()
print (strings_list)

# Open a file named "output.txt" in write mode
with open("user01.txt", "w") as file:
    # Write each string from the list to the file
    for string in strings_list:
        file.write(string + "\n")

