from distutils.log import debug
from fileinput import filename
import pandas as pd
from flask import *
import os
from werkzeug.utils import secure_filename
UPLOAD_FOLDER = os.path.join('staticFiles', 'uploads')
DOWNLOAD_FOLDER = os.path.join('results', 'd')
# Define allowed files
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER
 
app.secret_key = 'This is your secret key to utilize session in Flask'
"""# Anonymization parameters #""" 
k = 45           # K-anonymity
l = 2           # L-closeness
t = 0           # T-diversity
epsilon = 0.5   # Epsilon for Differential Privacy
delta = 0.001   # Delta for Differential Privacy


# Just to color the outputs
class Colors:
    """ ANSI color codes """
    BLACK           = "\033[0;30m"
    RED             = "\033[0;31m"
    GREEN           = "\033[0;32m"
    BROWN           = "\033[0;33m"
    BLUE            = "\033[0;34m"
    PURPLE          = "\033[0;35m"
    CYAN            = "\033[0;36m"
    LIGHT_GRAY      = "\033[0;37m"
    DARK_GRAY       = "\033[1;30m"
    LIGHT_RED       = "\033[1;31m"
    LIGHT_GREEN     = "\033[1;32m"
    YELLOW          = "\033[1;33m"
    LIGHT_BLUE      = "\033[1;34m"
    LIGHT_PURPLE    = "\033[1;35m"
    LIGHT_CYAN      = "\033[1;36m"
    LIGHT_WHITE     = "\033[1;37m"
    BOLD            = "\033[1m"
    FAINT           = "\033[2m"
    ITALIC          = "\033[3m"
    UNDERLINE       = "\033[4m"
    BLINK           = "\033[5m"
    NEGATIVE        = "\033[7m"
    CROSSED         = "\033[9m"
    END             = "\033[0m"




#
# Load Dataset
# ------------

@app.route('/', methods=['GET', 'POST'])
def uploadFile():
    if request.method == 'POST':
      # upload file flask
        f = request.files.get('file')
 
        # Extracting uploaded file name
        data_filename = secure_filename(f.filename)
 
        f.save(os.path.join(app.config['UPLOAD_FOLDER'],data_filename))
 
        session['uploaded_data_file_path'] =os.path.join(app.config['UPLOAD_FOLDER'],data_filename)
 
        return render_template('index2.html')
    return render_template("index.html")

@app.route('/show_data')
def showData():
    # Uploaded File Path
    dataset_path = session.get('uploaded_data_file_path', None)
    # read csv
    
    df = pd.read_csv(dataset_path,
                              encoding='unicode_escape')
    # Converting to html Table
    uploaded_df_html = df.head().to_html()
    
    return render_template('show_csv_data.html',
                           data_var=uploaded_df_html,len=len(df))

@app.route("/ins",methods=['GET', 'POST'])
def ins():
     dataset_path = session.get('uploaded_data_file_path', None)
    # read csv
    
     df = pd.read_csv(dataset_path,
                              encoding='unicode_escape')
     df.dropna(axis=0, inplace=True)
    
     stepCheck =  request.form["yes_no"]
     
     if (stepCheck == "no" ):
                         return render_template('error.html',e="[Error] Dataset parsing is wrong!")
     return render_template('ins.html',att=df.columns[0],columns=df.columns,lenp=len(df))
     
@app.route("/ml",methods=['GET', 'POST'])
def home():         
                    dataset_path = session.get('uploaded_data_file_path', None)
 
    
                    df = pd.read_csv(dataset_path,
                              encoding='unicode_escape')
                    df.dropna(axis=0, inplace=True)
                    attributes = {}
                   
                    for col in df.columns:
                            attributes[col] = {
                                'dataType': df[col].dtype,
                                'attributeType': ["Identifier", "Quasi-identifier", "Sensitive", "Insensitive"][int(request.form[col])-1]
                            }
                        
                            if (df[col].dtype.name == "object"):
                                       df[col] = df[col].astype("category")
                       


                    print(attributes)
                    # Making a copy of the dataset for the DP stats calculation
                    OrigDF = df.copy()



                    # Some datastructures for computational easiness
                    qi_index = list()
                    feature_columns = list()
                    sensitive_column = list()

                    for attribute in attributes:
                        if attributes[attribute]['attributeType'] == "Quasi-identifier":
                            feature_columns.append(attribute)
                            qi_index.append(list(OrigDF.columns).index(attribute))
                        elif attributes[attribute]['attributeType'] == "Sensitive":
                            sensitive_column.append(attribute)

                    feature_columns =  feature_columns if (len(feature_columns) > 0) else None
                    sensitive_column = sensitive_column[0] if (len(sensitive_column) > 0) else None



                    # 
                    # Predict Parameter Ranges
                    # ------------------------

                    from algorithms.param_predictor import ParamPredictor

                    res = (ParamPredictor()).predict(df, qi_index, sensitive_column)
                    print(f" - The nominal value for k is : {res['k']}")
                    resk=res['k']
                    print(f" - The l value should be within the range : [{res['l'][0]}, {res['l'][1]}]")
                    lran=[res['l'][0], res['l'][1]]
                    print(f" - The t value should be within the range : [{res['t']:.2f}, {0.0})")
                    tran=[res['t'], 0.0]






                    # ----------------------------------------------------------- Anonymization ------------------------------------------------------- #

                    #
                    # Supress direct identifiers with '*'
                    # -----------------------------------

                    for attribute in attributes:
                        if attributes[attribute]['attributeType'] == "Identifier":
                            df[attribute] = '*'



                    #
                    # Generalizing quasi-identifiers with k-anonymity
                    # -----------------------------------------------

                    from algorithms.anonymizer import Anonymizer


                    # Check if there are any quasi-identifiers
                    quasi = False
                    for attribute in attributes:
                        if attributes[attribute]['attributeType'] == "Quasi-identifier":
                            quasi = True


                    if not quasi:
                          return render_template("error.html",e="No Quasi-identifier found! At least 1 quasi-identifier is required.")

                    anon = Anonymizer(df, attributes)
                    anonymizedDF = anon.anonymize(k, l, t)

                    # Utility Measure
                    from utility.DiscernMetric import DM
                    from utility.CavgMetric import CAVG
                    from utility.GenILossMetric import GenILoss

                    qi_index = list()
                    for attribute in attributes:
                        if attributes[attribute]['attributeType'] == "Quasi-identifier":
                            qi_index.append(list(OrigDF.columns).index(attribute))




                    print("\n --------- Utility Metrices --------- \n")
                    # Discernibility Metric
                    raw_dm = DM(OrigDF, qi_index, k)
                    raw_dm_score = raw_dm.compute_score()

                    anon_dm = DM(anonymizedDF, qi_index, k)
                    anon_dm_score = anon_dm.compute_score()

                    print(f"DM score (lower is better): \n  BEFORE: {raw_dm_score} || AFTER: {anon_dm_score} || {raw_dm_score > anon_dm_score}")

                    # Average Equivalence Class
                    raw_cavg = CAVG(OrigDF, qi_index, k)
                    raw_cavg_score = raw_cavg.compute_score()

                    anon_cavg = CAVG(anonymizedDF, qi_index, k)
                    anon_cavg_score = anon_cavg.compute_score()

                    import math
                    print(f"CAVG score (near 1 is better): \n  BEFORE: {raw_cavg_score:.3f} || AFTER: {anon_cavg_score:.3f} || {math.fabs(1-raw_cavg_score) > math.fabs(1-anon_cavg_score)}")

                    # Gen I Loss Metric
                    GILoss = GenILoss(OrigDF, feature_columns)
                    geniloss_score = GILoss.calculate(anonymizedDF)

                    print(f"GenILoss: [0: No transformation, 1: Full supression] \n Value: {geniloss_score}")

                   

                    #
                    # Exporting data
                    # --------------

                    # anonymizedDF.to_csv(export_path+'.csv', index=False)
                    ch = request.form["ad"]
                    if not (ch == 'yes'):
                        return render_template('out.html',nom=resk,lran=lran,tran=tran,raw_dm_score=raw_dm_score,anon_dm_score=anon_dm_score,firstb=(raw_dm_score > anon_dm_score),
                                           secondb=(math.fabs(1-raw_cavg_score) > math.fabs(1-anon_cavg_score)),raw_cavg_score=raw_cavg_score,anon_cavg_score=anon_cavg_score,geniloss_score=geniloss_score,val=not (ch == 'yes'))


                    export_path = "AnonymizedData"
                    print("\nExporting anonymized dataset ... ")


                    # Create a Pandas Excel writer object using XlsxWriter as the engine.
                    writer = pd.ExcelWriter("results/d/"+export_path + '.xlsx')


                    qi_index = list()
                    for attribute in attributes:
                        if attributes[attribute]['attributeType'] == "Quasi-identifier":
                            qi_index.append(list(OrigDF.columns).index(attribute))


                    def paint_bg(v, color):
                        ret = [f"background-color: {color[0]};" for i in v]
                        return ret

                    anonymizedDF = anonymizedDF.style.hide().apply(paint_bg, color=['gainsboro', 'ivory'], axis=1) 


                    # Write a dataframe to the worksheet.
                    anonymizedDF.to_excel(writer, sheet_name ='Data', index=False)
                    # DP_out.to_excel(writer, sheet_name ='Stats', index=False)

                    # print(DP_out)

                    # Close the Pandas Excel writer object and output the Excel file.
                    writer.close()
                    session['fname'] =(export_path + '.xlsx')
                    return render_template('out.html',nom=resk,lran=lran,tran=tran,raw_dm_score=raw_dm_score,anon_dm_score=anon_dm_score,firstb=(raw_dm_score > anon_dm_score),
                                           secondb=(math.fabs(1-raw_cavg_score) > math.fabs(1-anon_cavg_score)),raw_cavg_score=raw_cavg_score,anon_cavg_score=anon_cavg_score,geniloss_score=geniloss_score,val=not (ch == 'yes'))

@app.route("/download", methods=['GET', 'POST'])
def download():
    print(app.config["DOWNLOAD_FOLDER"]+session.get('fname'))
    return send_from_directory(directory=app.config["DOWNLOAD_FOLDER"], path=session.get('fname'), as_attachment=True)
if __name__ == '__main__':
    app.run(debug=True)