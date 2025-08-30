import statistics
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import requests
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# INPUT IN TWO SCENARIO NAMES (first is X axis)
SCENARIO_NAMES = ['VT Air Advanced', 'Air Voltaic']

# INPUT SET STEP SIZE FOR GRID ON GRAPH
STEP = [100, 4]

# INPUT AMOUNT OF LEADERBOARD PAGES TO PULL. EACH PAGE HAS 100 ENTRIES, SO 1 ITERATES THROUGH 200 ENTRIES (page 0,and 1)
MAX_PAGES = [5, 5]

# IF YOU WANT TO SEE MAX SCORES PUT "Y" BELOW
Show_Max = "Y"

# USE MEDIAN TO IDENTIFY TRENDLINE PUT "Y" BELOW (ELSE MAX SCORES ARE USED)
UseMedian = "N"

# ARRAY SETUP
Leaderboard_ID = [0] * len(SCENARIO_NAMES)

# REQUEST SCENARIO PATH ONE TIME TO GET AMOUNT OF PAGES ON THE SCENARIOS PAGE
session = requests.Session()
r = session.get("https://kovaaks.com/webapp-backend/scenario/popular?page=0&max=100").json()
Max_Page = r['total']//100

# ITERATE THROUGH ALL PLAYLIST PAGES
for i in range(Max_Page + 1):
    r = session.get(f"https://kovaaks.com/webapp-backend/scenario/popular?page={i}&max=100").json()

    # ITERATE THROUGH ALL "data" ROWS ON EACH PLAYLIST PAGE
    for Data in r['data']:

        # IF SCENARIO NAME IS FOUND FILL THE CORRESPONDING INDEX IN THE LEADERBOARD ID ARRAY WITH THE "leaderboardId"
        try:
            index = SCENARIO_NAMES.index(Data['scenarioName'])
            Leaderboard_ID[index] = Data['leaderboardId']
            print(f"Scenario ID Found for: {SCENARIO_NAMES[index]}, {Leaderboard_ID[index]}")
        except ValueError:
            pass

    # EXIT LOOP IF ALL LEADERBOARD IDs HAVE BEEN FOUND
    if all(value != 0 for value in Leaderboard_ID):
        break
session.close()

# CREATE DICTIONARY
Score_Dic = {}

# ITERATE THROUGH EACH LEADERBOARD
Counti = 0
for i in range(0, len(SCENARIO_NAMES)):

    Name = SCENARIO_NAMES[i]
    Name = Name.replace(" ", "+")

    # REQUEST LEADERBOARD PATH ONE TIME TO GET AMOUNT OF PAGES ON EACH LEADERBOARD
    session = requests.Session()
    r = session.get(f"https://kovaaks.com/webapp-backend/leaderboard/scores/global?leaderboardId={Leaderboard_ID[i]}&page=0&max=100").json()
    Max_Page = r['total']//100

    # MANUAL LIMIT
    Max_Page = min(Max_Page,MAX_PAGES[i])

    # ITERATE THROUGH ALL LEADERBOARD PAGES
    for ii in range(Max_Page + 1):
        r = session.get(f"https://kovaaks.com/webapp-backend/leaderboard/scores/global?leaderboardId={Leaderboard_ID[i]}&page={ii}&max=100").json()
        print(f"Leaderboard {i + 1} of {len(SCENARIO_NAMES)}. Page: {ii} of {Max_Page} data pull.")


        # ITERATE THROUGH ALL "data" ROWS ON EACH PLAYLIST PAGE AND SEND DATA TO LEADERBOARD COLUMN OF RELEVANT ARRAYS
        for Data in r['data']:

            # IF PERSON HAS AN ACTIVE KOVAAK PROFILE
            if 'webappUsername' in Data and Data['webappUsername'] is not None:

                # GET ACCOUNT ID AND NAME
                try:
                    Steam_ID = Data['steamAccountName']
                    Kovaak_Name = Data['webappUsername']
                    Kovaak_Name2 = Kovaak_Name.replace(" ", "+")
                    MaxScore = Data['score']

                    #print(Kovaak_Name2)
                    # ITERATE TO USERS LAST 10 PLAY PAGE
                    r2 = session.get(f"https://kovaaks.com/webapp-backend/user/scenario/last-scores/by-name?username={Kovaak_Name2}&scenarioName={Name}").json()

                    # ITERATE THROUGH ALL SCORES IN TOP 10 SCORE
                    epoch = 0
                    count = 0
                    scores = []

                    date_time1 = None
                    for entry in r2:

                        # IF NO ERRORS
                        if "error" not in entry:
                            score = entry["score"]
                            epoch = entry["attributes"]["epoch"]
                            epoch = int(epoch)

                            #CONVERT THE EPOCH TO DATE TIME
                            epoch = epoch / 1000  # Convert to seconds
                            date_time = datetime.fromtimestamp(epoch)  # Convert to datetime
                            if date_time1 == None:
                                date_time1 = date_time

                            # IF PLAY IS WITHIN 1 YEAR OF MOST RECENT PLAY
                            difference = date_time1 - date_time
                            daysd = difference.days

                            # IF SCORE IS WITHIN 1 YEAR OF FIRST SCORE
                            if daysd < 365:
                                count += 1
                                scores.append(score)

                        # GET ACTUAL SCORE AVERAGE
                        if count > 3:
                            Final_Score = statistics.median(scores)

                            # IF STEAM NAME (KEY) EXISTS FILL IN RELEVANT SCORE LIST FOR STEAM NAME
                            if Steam_ID in Score_Dic:
                                Score_Dic[Steam_ID][Counti] = Final_Score
                                Score_Dic[Steam_ID][Counti + 1] = date_time1
                                Score_Dic[Steam_ID][Counti + 2] = MaxScore

                            # IF STEAM NAME (KEY) DOES NOT EXIST, CREATE NEW KEY FOR STEAM NAME AND FILL IN RELEVANT SCORE LIST FOR STEAM NAME
                            elif Steam_ID not in Score_Dic:
                                Score_Dic[Steam_ID] = [None]*len(SCENARIO_NAMES)*3
                                Score_Dic[Steam_ID][Counti] = Final_Score
                                Score_Dic[Steam_ID][Counti + 1] = date_time1
                                Score_Dic[Steam_ID][Counti + 2] = MaxScore
                except KeyError:
                    pass
    session.close()
    Counti += 3

# ITERATE THROUGH ALL VALUES IN THE DICTIONARY AND APPEND GOOD VALUES TO ARRAY FOR PLOTTING
array1 = []
array2 = []
array3 = []
array4 = []

for key, value in Score_Dic.items():

    if value[1] is not None and value[4] is not None:
        Date1 = value[1]
        Date2 = value[4]
        difference = Date1 - Date2
        daysd = difference.days

        if abs(daysd) < 365 and value[0] is not None and value[3] is not None:

            if value[0] != 0 and value[3] != 0 and (value[2] - value[0]) != 0:
                # MEDIAN
                array1.append(value[0])
                array2.append(value[3])
                # MAX
                array3.append(value[2])
                array4.append(value[5])


# PLOTTING AND EXPORT OF PLOT
array1 = np.array(array1)  # Ensure array1 is a NumPy array
array2 = np.array(array2)  # Ensure array2 is a NumPy array
array3 = np.array(array3)  # Ensure array1 is a NumPy array
array4 = np.array(array4)  # Ensure array2 is a NumPy array

# Calculate the appropriate limits for x-axis and y-axis to ensure multiples of the step size
x_min = np.floor(min(array1) / STEP[0]) * STEP[0]
x_max = np.ceil(max(array3) / STEP[0]) * STEP[0]

y_min = np.floor(min(array2) / STEP[1]) * STEP[1]
y_max = np.ceil(max(array4) / STEP[1]) * STEP[1]

fig1 = plt.figure(figsize=(10, 10))
plt.scatter(array1, array2, color='black', marker='*', label='Median')

if Show_Max == "Y":
    plt.scatter(array3, array4, color='blue', marker='*', label='Max')
    for i in range(len(array1)):
        plt.plot([array1[i], array3[i]], [array2[i], array4[i]], color='gray', linewidth=0.5)

# FIT A TREND LINE 1 DEGREE POLYNOMINAL (TREND LINE IS BAD) SET UP THE WEIGHTS
if UseMedian == "Y":
    t=2
else:
    array1 = array3
    array2 = array4


# Parameters
degree = 1  # Linear
std_thresholds = [2, 1,0.5]  # Filter progressively

# Start with all points
x = array1.copy()
y = array2.copy()

# Iterative filtering
for thresh in std_thresholds:
    coeffs = np.polyfit(x, y, degree)
    fit = np.polyval(coeffs, x)
    residuals = y - fit
    std_resid = np.std(residuals)
    mask = np.abs(residuals) <= thresh * std_resid
    x, y = x[mask], y[mask]

# Final fit after filtering
coeffs = np.polyfit(x, y, degree)
trend_line_x = np.linspace(x.min(), x.max(), 100)
trend_line_y = np.polyval(coeffs, trend_line_x)

# Equation parameters
slope, intercept = coeffs
equation = f"y = {slope:.4f}x + {intercept:.4f}"

# R² using actual data
model = LinearRegression()
model.fit(array1.reshape(-1, 1), array2)
predicted = model.predict(array1.reshape(-1, 1))
r2 = r2_score(array2, predicted)


x_line = np.array([0, x_max])
y_line = slope * x_line + intercept

plt.plot(x_line, y_line, color='red', linewidth=1)

# FINAL PLOTTING STUFF
plt.title(SCENARIO_NAMES[1] + ' vs ' + SCENARIO_NAMES[0] + ', Recent Median Score Correlation')
fig1.suptitle(f"Equation of the trendline: y = {slope:.4f}x + {intercept:.4f}   R² value: {r2:.4f}", y=0.92, fontsize=12)
plt.xlim(None, None)
plt.ylim(None, None)
plt.grid(True, linestyle='--', color='gray', alpha=0.6)
plt.xlabel(SCENARIO_NAMES[0] + ' Scores')
plt.ylabel(SCENARIO_NAMES[1] + ' Scores')


# Set limits for both axes
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

# Set x-ticks and y-ticks to show every step_size units
plt.xticks(np.arange(x_min, x_max + STEP[0], STEP[0]))
plt.yticks(np.arange(y_min, y_max + STEP[1], STEP[1]))

plt.legend()
fig1.savefig(SCENARIO_NAMES[1] + " vs " + SCENARIO_NAMES[0] +".png")

print(f"Equation of the trendline: y = {slope:.4f}x + {intercept:.4f}")
print(f"R² value: {r2:.4f}")

