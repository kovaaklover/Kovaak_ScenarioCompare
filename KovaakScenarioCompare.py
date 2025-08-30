# USER INPUTS
SCENARIO_NAMES = ['VT Air Advanced', 'Air Voltaic']     # INPUT IN THE TWO SCENARIOS YOU WANT TO COMPARE (first is X axis)
STEP = [100, 4]     # INPUT SET STEP SIZE FOR GRID ON GRAPH
MAX_ENTRIES = [600, 600]    # INPUT AMOUNT OF LEADERBOARD ENTRIES TO PULL

# IMPORTS
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import requests

# ARRAY SETUP
Leaderboard_ID = [0] * len(SCENARIO_NAMES)
Play_Avg = [0] * len(SCENARIO_NAMES)

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
            Plays = Data['counts']['plays']
            Entries = Data['counts']['entries']
            Play_Avg[index] = Plays/Entries
        except ValueError:
            pass

    # EXIT LOOP IF ALL LEADERBOARD IDs HAVE BEEN FOUND
    if all(value != 0 for value in Leaderboard_ID):
        break
session.close()

# CREATE DICTIONARY
Score_Dic = {}

# FUNCTION TO PROCESS EACH PAGE OF EACH LEADERBOARD (FUNCTION CALLED VIA THREADING)
def process_leaderboard(leaderboard_id, page, session, Counti, score_lock, Score_Dic,Name, LeaderboardCount, Max_Page, i):

    # PULL LEADERBOARD
    r = session.get(f"https://kovaaks.com/webapp-backend/leaderboard/scores/global?leaderboardId={leaderboard_id}&page={page}&max=10").json()

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
                    with score_lock:
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

    print(f"Leaderboard {i + 1} of {LeaderboardCount}. Page: {page} of {Max_Page} data pull.")

# THREADING AND LOCK PROTECTION
score_lock = Lock()  # Create a lock for protecting shared resources

# START THREADER
with requests.Session() as session:  # Create ONE session
    with ThreadPoolExecutor(max_workers=100) as executor:
        Counti = 0
        futures = []

        # ITERATE THROUGH ALL LEADERBOARDS
        for i in range(len(Leaderboard_ID)):

            # REQUEST LEADERBOARD PATH ONE TIME TO GET AMOUNT OF PAGES ON EACH LEADERBOARD
            r = session.get(f"https://kovaaks.com/webapp-backend/leaderboard/scores/global?leaderboardId={Leaderboard_ID[i]}&page=0&max=10").json()
            Max_Entries = r['total']

            # MANUAL LIMIT
            Max_Entries = min(Max_Entries, MAX_ENTRIES[i])
            Max_Page = Max_Entries // 10

            # SCENARIO NAME
            Name = SCENARIO_NAMES[i]
            Name = Name.replace(" ", "+")

            LeaderboardCount = len(SCENARIO_NAMES)

            # ITERATE THROUGH ALL LEADERBOARD PAGES AND SEND TO FUNCTION
            for ii in range(Max_Page):
                futures.append(executor.submit(process_leaderboard, Leaderboard_ID[i], ii, session, Counti, score_lock,Score_Dic, Name, LeaderboardCount, Max_Page, i))

            # LOCK CRITERIA (NEEDED)
            with score_lock:
                Counti += 3

        # PROCESS RESULTS
        for future in as_completed(futures):
            future.result()  # No need to handle this since the processing is done within the function

# ITERATE THROUGH ALL VALUES IN THE DICTIONARY AND APPEND GOOD VALUES TO ARRAY FOR PLOTTING
array1 = []
array2 = []
array3 = []
array4 = []

for key, value in Score_Dic.items():

    # IF PLAYER PLAYED BOTH SCENARIOS
    if value[1] is not None and value[4] is not None:
        Date1 = value[1]
        Date2 = value[4]
        difference = Date1 - Date2
        daysd = difference.days

        # IF PLAYER PLAYED BOTH WITHIN ONE YEAR OF EACH OTHER
        if abs(daysd) < 365 and value[0] is not None and value[3] is not None:

            # I FORGOT WHY I HAD THIS???
            if value[0] != 0 and value[3] != 0 and (value[2] - value[0]) != 0:

                # MEDIAN SCORE ARRAY
                array1.append(value[0])
                array2.append(value[3])

                # MAX SCORE ARRAY
                array3.append(value[2])
                array4.append(value[5])

# CONVERT DATA TO NUMPY ARRAYS
array1 = np.array(array1)  # Ensure array1 is a NumPy array
array2 = np.array(array2)  # Ensure array2 is a NumPy array
array3 = np.array(array3)  # Ensure array1 is a NumPy array
array4 = np.array(array4)  # Ensure array2 is a NumPy array

# SET PLOT LIMITS
x_min = np.floor(min(array1) / STEP[0]) * STEP[0]
x_max = np.ceil(max(array3) / STEP[0]) * STEP[0]
y_min = np.floor(min(array2) / STEP[1]) * STEP[1]
y_max = np.ceil(max(array4) / STEP[1]) * STEP[1]


# FUNCTION FOR FILTERED TRENDLINE
def filtered_trendline(x, y, degree, std_thresholds):

    # ITERATE THROUGH STD
    for thresh in std_thresholds:
        coeffs = np.polyfit(x, y, degree)
        fit = np.polyval(coeffs, x)
        residuals = y - fit
        std_resid = np.std(residuals)
        mask = np.abs(residuals) <= thresh * std_resid
        x, y = x[mask], y[mask]

    # Final polynomial fit
    coeffs = np.polyfit(x, y, degree)
    return coeffs

# MEDIAN TRENDLINE
x = array1.copy()
y = array2.copy()
coeffs = filtered_trendline(x, y, degree=1, std_thresholds=[2, 1.5, 1])
slope1, intercept1 = coeffs

# MAX TRENDLINE
x = array3.copy()
y = array4.copy()
coeffs = filtered_trendline(x, y, degree=1, std_thresholds=[2, 1.5, 1])
slope2, intercept2 = coeffs

# PLOT MEDIAN AND MAX POINTS
fig1 = plt.figure(figsize=(10, 10))
plt.scatter(array1, array2, color='black', marker='*', label='Median')
plt.scatter(array3, array4, color='blue', marker='*', label='Max')
for i in range(len(array1)):
    plt.plot([array1[i], array3[i]], [array2[i], array4[i]], color='gray', linewidth=0.5)

# PLOTTING TRENDLINES
x_line = np.array([0, x_max])
y_line1 = slope1 * x_line + intercept1
y_line2 = slope2 * x_line + intercept2
plt.plot(x_line, y_line1, color='red', linewidth=1, label='Median Trend Line')
plt.plot(x_line, y_line2, color='red', linewidth=1, linestyle=':', label='Max Trend Line')

# FINAL PLOTTING STUFF
plt.title(f"{SCENARIO_NAMES[1]} vs {SCENARIO_NAMES[1]} Recent Score Correlation: Based on top {len(x)} players")
fig1.suptitle(
    f"Equation of the median score trendline: y = {slope1:.4f}x + {intercept1:.4f}\n"
    f"Equation of the max score trendline: y = {slope2:.4f}x + {intercept2:.4f}",
    y=0.98,
    fontsize=12
)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.grid(True, linestyle='--', color='gray', alpha=0.6)
#plt.xlabel(f"{SCENARIO_NAMES[0]} Scores  (Avg plays per person {float(Play_Avg[0]):,.1f})")
#plt.ylabel(f"{SCENARIO_NAMES[1]} Scores  (Avg plays per person {float(Play_Avg[1]):,.1f})")
plt.xlabel(f"{SCENARIO_NAMES[0]} Scores")
plt.ylabel(f"{SCENARIO_NAMES[1]} Scores")

# SET X AND Y TICKS
plt.xticks(np.arange(x_min, x_max + STEP[0], STEP[0]))
plt.yticks(np.arange(y_min, y_max + STEP[1], STEP[1]))

plt.legend()
plt.tight_layout()
fig1.savefig(SCENARIO_NAMES[1] + " vs " + SCENARIO_NAMES[0] +".png")

print(f"Equation of the median score trendline: y = {slope1:.4f}x + {intercept1:.4f}")
print(f"Equation of the max score trendline: y = {slope2:.4f}x + {intercept2:.4f}")