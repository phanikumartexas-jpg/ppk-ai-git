import pandas as pd;
import matplotlib.pyplot as pip
import os as os;
import gc as gc;
import numpy as np;

import sklearn.preprocessing as preprocessing
import sklearn.linear_model as linear_model
import sklearn.model_selection as model_selection
import sklearn.metrics as metrics
import joblib as joblib

class MyClass :
  __logCtr:int = 0;
  __logTime:str = None;
  __empDataClientFile:str = None
  __empDetailsDataClientFile:str = None
  __departmentDataClient:str = None

  def __init__(self) -> None:
    MyClass.__empDataClientFile = "AIG-Employee-Retention\dataset\employee_data_client.csv"
    MyClass.__empDetailsDataClientFile = "AIG-Employee-Retention\dataset\employee_details_data_client.csv"
    MyClass.__departmentDataClient = "AIG-Employee-Retention\dataset\department_data_client.csv"

    pd.options.display.float_format = '{:.3f}'.format
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    return    


  @classmethod
  def println(cls, pA:object, pB:object = None) -> None:
    MyClass.__logCtr += 1
    MyClass.__logTime = pd.Timestamp.now().strftime(format="%m/%d/%y %H:%M:%S")
    if (pB is None) :
      print(f"{MyClass.__logTime} {MyClass.__logCtr} ) ----- {pA} -----\n\n")
    else:
      print(f"{MyClass.__logTime} {MyClass.__logCtr} ) ----- {pA} -----\n{pB}\n\n")
    return

  def readFiles(self)-> tuple[pd.DataFrame,pd.DataFrame, pd.DataFrame]:
    empData:pd.DataFrame = pd.read_csv(MyClass.__empDataClientFile)
    empDetailsData:pd.DataFrame = pd.read_csv(MyClass.__empDetailsDataClientFile)
    deptData:pd.DataFrame = pd.read_csv(MyClass.__departmentDataClient)

    MyClass.println("EmpData",empData.head(20))
    MyClass.println("EmpDetails", empDetailsData.head(20))
    MyClass.println("deptData", deptData.head(20))

    MyClass.prepareStats(empData)
    MyClass.prepareStats(empDetailsData)
    MyClass.prepareStats(deptData)
    return empData,empDetailsData,deptData

  @classmethod
  def prepareStats(self,pDF:pd.DataFrame) -> pd.DataFrame:
    vStatsDF:pd.DataFrame = pd.DataFrame(data=pDF.describe().T, index=pDF.columns.values)
    vStatsDF["type"] = pDF.dtypes.values
    vStatsDF["nulls"] = pDF.isna().sum()
    vStatsDF["notNulls"] = pDF.notna().sum()
    vStatsDF["totUniques"] = [pDF[vCol].nunique() for vCol in pDF.columns.values]
    vStatsDF["uniques"] = ["> 5" if pDF[vCol].nunique() > 5 else pDF[vCol].unique() for vCol in pDF.columns.values]
    MyClass.println(f"Stats",vStatsDF.sort_values(by="type",ascending=True))
    return vStatsDF; 

  def cleanData(self,pEmpDF:pd.DataFrame, pEmpDataDF:pd.DataFrame, pDeptDF:pd.DataFrame) -> tuple [pd.DataFrame, pd.DataFrame, pd.DataFrame] :
    # Fixing empDataDF
    # Remove NaN for Filed_complaint
    resultsetIdx = pEmpDF.query("filed_complaint.isna() == True").index
    pEmpDF.loc[resultsetIdx,"filed_complaint"] = pEmpDF["filed_complaint"].fillna(0)
    resultsetIdx = pEmpDF.query("filed_complaint.isna() == True").index
    MyClass.prepareStats(pEmpDF)

    # Fix NaN recently_promoted feature
    resultsetIdx = pEmpDF.query("recently_promoted.isna() == True").index
    pEmpDF.loc[resultsetIdx,"recently_promoted"] = pEmpDF["recently_promoted"].fillna(0)
    resultsetIdx = pEmpDF.query("recently_promoted.isna() == True").index
    MyClass.prepareStats(pEmpDF)

    # Fix Nans for Tenure
    resultset = pEmpDF["tenure"].value_counts(normalize=True)
    resultsetIdx = pEmpDF.query("tenure.isna() == True").index
    pEmpDF.loc[resultsetIdx,"tenure"] = np.random.choice(resultset.index,p=resultset.values,size=pEmpDF["tenure"].isna().sum())
    MyClass.prepareStats(pEmpDF)

    # Fix nulls for last_evaluation
    resultsetIdx = pEmpDF.query("last_evaluation.isna() == True").index
    # resultset = pEmpDF["last_evaluation"].round(2).value_counts(normalize=True)
    # pEmpDF.loc[resultsetIdx,"last_evaluation"] = np.random.choice(resultset.index, p=resultset.values, size=pEmpDF["last_evaluation"].isna().sum())
    pEmpDF = pEmpDF.drop(resultsetIdx,axis=0)

    # Fix nulls for satisfaction
    resultsetIdx = pEmpDF.query("satisfaction.isna() == True").index
    resultset = pEmpDF["satisfaction"].round(2).value_counts(dropna=True,normalize=True);
    pEmpDF.loc[resultsetIdx,"satisfaction"] = np.random.choice(resultset.index, p=resultset.values, size=pEmpDF["satisfaction"].isna().sum())

    # Fixing department nulls
    resultsetIdx = pEmpDF.query("department.isna() == True").index
    resultset = pEmpDF["department"].value_counts(normalize=True)
    MyClass.println("department resultset ", resultset)
    pEmpDF.loc[resultsetIdx,"department"] = np.random.choice(resultset.index,p=resultset.values,size=pEmpDF["department"].isna().sum())

    # replace department -IT with D00-IT
    resultsetIdx = pEmpDF.query("department == '-IT'").index
    pEmpDF.loc[resultsetIdx,"department"] = "D00-IT"
    MyClass.prepareStats(pEmpDF)  
    MyClass.prepareStats(pEmpDataDF)
    MyClass.prepareStats(pDeptDF)
     
    return pEmpDF, pEmpDataDF, pDeptDF
  
  def prepareMasterDF(self,pEmpDF:pd.DataFrame, pEmpDetailsDF:pd.DataFrame, pDeptDF:pd.DataFrame) -> pd.DataFrame :
    vDF:pd.DataFrame = pEmpDF.merge(right=pEmpDetailsDF,how="left",left_on="employee_id", right_on="employee_id")
    vDF:pd.DataFrame = vDF.merge(right=pDeptDF,how="left",left_on="department",right_on="dept_id")
    MyClass.println("The combination table is ", vDF.sample(10))
    return vDF

  def cleanupMasterData (self,pDF:pd.DataFrame) -> pd.DataFrame :
    MyClass.println("pDF Records", pDF.sample(10))

    # drop temporary Employees
    resultsetIdx = pDF.query("department == 'D00-TP'").index
    pDF = pDF.drop(resultsetIdx)

    resultsetIdx = pDF.query("employee_id == 0").index
    MyClass.println("Zero employee records", pDF.loc[resultsetIdx])
    pDF = pDF.drop(resultsetIdx)
    pDF = pDF.drop_duplicates()
    pDF = pDF.drop_duplicates(subset=["employee_id"],keep="last")


    resultsetIdx = pDF.query("employee_id == 0").index
    MyClass.println("Zero employee records", pDF.loc[resultsetIdx])

    pDF = pDF.drop(["employee_id","dept_id","dept_name","dept_head"],axis=1)
    MyClass.println("Modified Combined DF", pDF.head(10))


    pDF = pDF.astype({"filed_complaint":"str",
                      "recently_promoted":"str"})
    statsDF = MyClass.prepareStats(pDF)
    MyClass.println("The stats of final dataframe", statsDF)    
    return pDF

  def inputOutputFeatures(self, pDF:pd.DataFrame) -> tuple[pd.DataFrame, pd.Series] :
    X = pDF.drop(columns=["status"])
    y = pDF["status"]
    return X,y
  
  def eda(self,pDF:pd.DataFrame) -> None :
    result =  pDF.corr(method="pearson")
    MyClass.println("Correlation is ", result)
    np.fill_diagonal(result.values,0)
    MyClass.println("The flat stack is ", result)
    maxCorrVal = result.stack().max()
    MyClass.println("The Max value is", maxCorrVal)
    maxIndex = result.stack().idxmax()
    MyClass.println("The Max Index is", maxIndex)    

    return 
  
  def splitTestAndTrain(self,pX:pd.DataFrame, py:pd.DataFrame) -> tuple [pd.DataFrame, pd.DataFrame, pd.Series, pd.Series] :
    vXTrain:pd.DataFrame = None
    vyTrain:pd.Series = None
    vXTrain, vXTest, vyTrain, vyTest = model_selection.train_test_split(pX, py, test_size=0.2)
    MyClass.println(f"vXTrain size = {vXTrain.shape}", vXTrain.head(10))
    MyClass.println(f"vyTrain size = {vyTrain.shape}", vyTrain.head(10))
    return vXTrain, vXTest, vyTrain, vyTest;

  def splitContinuousVsCategorical(self, pX:pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame] :
    vXCont:pd.DataFrame = None
    vXCat:pd.DataFrame = None

    vXCont = pX.select_dtypes(exclude="object")
    vXCat = pX.select_dtypes(include="object")    

    MyClass.println("The Continous Size = {vXCont.shape}", vXCont.head(10))
    MyClass.println("The Continous Size = {vXCat.shape}", vXCat.head(10)) 
    return vXCont, vXCat;

  def scaleContinousData (self,pDF:pd.DataFrame) -> pd.DataFrame :
    standardScalar:preprocessing.StandardScaler = preprocessing.StandardScaler()
    standardScalar.fit(pDF)
    result = standardScalar.transform(pDF)
    vScale:pd.DataFrame = pd.DataFrame(data=result,index=pDF.index,columns=pDF.columns.values)
    MyClass.println("The scaled data is ", vScale.head(10))
    return vScale
  
  def encodingCategoricalData(self,pDF:pd.DataFrame) -> pd.DataFrame :
    oneHotEncoder:preprocessing.OneHotEncoder = preprocessing.OneHotEncoder(sparse_output=False,drop="first")
    oneHotEncoder.fit(pDF)
    result = oneHotEncoder.transform(pDF)
    vEncoder = pd.DataFrame(data=result,index=pDF.index,columns=oneHotEncoder.get_feature_names_out())
    MyClass.println("Encoder",vEncoder.head(10))
    return vEncoder
  
  def combineToPrepareModelData(self,pCont:pd.DataFrame, pCat:pd.DataFrame) -> pd.DataFrame :
    vDF:pd.DataFrame = pd.concat([pCont,pCat],axis=1)
    MyClass.println("Model Combined Data",vDF.head(10))
    MyClass.println("Model Combined columns",vDF.columns.values)
    return vDF
  
  def convertYtoOneAndZero(self,pY:pd.Series ) -> pd.Series :
    vY:pd.Series = np.where(pY == "Employed",0,1)
    MyClass.println("converted vY", vY)
    return vY;
  
  def createTheModel(self,pX:pd.DataFrame, py:pd.DataFrame) -> linear_model.LogisticRegression :
    vModel:linear_model.LogisticRegression = linear_model.LogisticRegression(penalty="l2",C=1.0)
    vModel.fit(pX,py)
    return vModel
  
  def testModel(self,pModel:linear_model.LogisticRegression,pX:pd.DataFrame) -> pd.DataFrame :
    outcome:pd.DataFrame = None
    resultset:pd.Series = pModel.predict_proba(pX)
    MyClass.println("The result predicted proba ", resultset)
    outcome = pd.DataFrame(data=resultset, index=pX.index, columns=["0-Prediction","1-Prediction"])
    return outcome;

  def findModelThreshold(self,vA:pd.Series,vP:pd.DataFrame) -> float :
    vP["actuals"] = vA
    th:float = 0.5
    thList:list = np.linspace(start=0.025,stop=0.975,num=39)
    MyClass.println("The predicted and actuals are ", vP.head(10))
    thSeries:pd.Series = pd.Series(index=thList)

    for vTh in thList :
      vP[vTh] = np.where(np.round(vP["1-Prediction"],3) < vTh,0,1)
      thSeries[vTh] = metrics.f1_score(y_true=vP["actuals"],y_pred=vP[vTh])
    
    MyClass.println("The threshhold scores are ",thSeries)
    th = thSeries.idxmax();
    MyClass.println("The Max threshold is ", th)
    # MyClass.println("The threshhold result is ", vP.head(20))    
    return th;

  def generatePrediction(self,pThreshold:float,py:pd.DataFrame) -> pd.Series :
    resultset = np.where(np.round(py["1-Prediction"],3) >= np.round(pThreshold,3),"Left","Employed")
    pOutput:pd.Series = pd.Series(resultset,index=py.index)
    # MyClass.println("The output prediction is ", pOutput)
    return pOutput
  
  def generateOutput(self,pX:pd.DataFrame,py:pd.Series, pyp:pd.Series) -> pd.DataFrame :
    vOutput:pd.DataFrame = pd.concat([pX,py,pyp],axis=1)
    MyClass.println("The result of the Model is ", vOutput.sample(20))
    return vOutput

  def main(self) -> None:
    empDF, empTrxDF, deptDF = self.readFiles()
    empDF, empTrxDF, deptDF = self.cleanData(empDF, empTrxDF, deptDF)
    df = self.prepareMasterDF(empDF, empTrxDF, deptDF)

    empDF = None
    empTrxDF = None
    deptDF = None
    gc.collect()     

    df = self.cleanupMasterData(df)  
    X,y = self.inputOutputFeatures(df)
    XTrain, XTest, yTrain, yTest = self.splitTestAndTrain(X,y)
    XTrainCon, XTrainCat = self.splitContinuousVsCategorical(XTrain)
    XTestCon, XTestCat = self.splitContinuousVsCategorical(XTest)

    self.eda(XTrainCon)
    XTrainScale:pd.DataFrame = self.scaleContinousData(XTrainCon)
    XTrainCat:pd.DataFrame = self.encodingCategoricalData(XTrainCat)
    
    XTrainModelDataCombine:pd.DataFrame = self.combineToPrepareModelData(XTrainScale,XTrainCat)
    yTrainActuals:pd.Series = self.convertYtoOneAndZero(yTrain);

    model:linear_model.LogisticRegression = self.createTheModel(XTrainModelDataCombine,yTrainActuals)
    yTrainPrediction:pd.DataFrame = self.testModel(model,XTrainModelDataCombine)
    modelThreshold:float = self.findModelThreshold(yTrainActuals,yTrainPrediction)
    yTrainPrediction:pd.Series = self.generatePrediction(modelThreshold, yTrainPrediction)
    generateOutput:pd.DataFrame = self.generateOutput(XTrain,yTrain,yTrainPrediction)

    return

if (__name__ == "__main__") :
  os.system("cls")
  MyClass.println("Execution started ..")
  obj:MyClass = MyClass()
  obj.main()