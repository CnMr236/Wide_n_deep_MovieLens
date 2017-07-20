#Tensorflow Simulationsstudie in R

#Parameters
vTrainTest=1:5
vMultiplikator=1

mSimulations = expand.grid(vTrainTest,vMultiplikator)
mSimulations = cbind(1:nrow(mSimulations),mSimulations)

#load Dataset
load("Database_ML100k.Rdata")

#In Dataframes umwandeln 
userdata = data.frame(mRatings, row.names = NULL)
testuser1 = data.frame(mTest1, row.names = NULL)  
testuser2 = data.frame(mTest2, row.names = NULL) 
testuser3 = data.frame(mTest3, row.names = NULL) 
testuser4 = data.frame(mTest4, row.names = NULL) 
testuser5 = data.frame(mTest5, row.names = NULL) 

#Alle Testdaten zusammenführen
totaltest <- rbind(testuser1,testuser2,testuser3,testuser4,testuser5)

#Testdaten - nicht relevante Item-Spalten ausblenden
testusernew <- totaltest[,-(4:502)]
testusertotal <- testusernew[,-(2)]

#Ausblenden der Test-Items in Userdata
Trainingset <- userdata[!(userdata$UserID %in% testusertotal$UserID & userdata$ItemID %in% testusertotal$ItemID.Rank),]

#Write Data - Input für Deep Learning model
write.csv(Trainingset, "Data/userdata.csv")

#TF-Model aufrufen / Trainieren
library(tensorflow)
Sys.setenv(TENSORFLOW_PYTHON="/usr/local/bin/python")
Sys.setenv(TENSORFLOW_PYTHON="source ~/tensorflowC/bin/activate")
TFTrain <- system("python DNNRTrain.py", wait = TRUE)

mResult=NULL
#Simulationsschleife Test Datensatz
for(iSimulation in 1:nrow(mSimulations)){
  
  # Laden der Parameter f?r diesen Simulationslauf
  vSimulation=as.numeric(mSimulations[mSimulations[,1]==iSimulation,])
  
  # Aktuellen Testdatensatz spezifizieren
  # ID des Testdatensatzes ist ein Parameter der Simulationsstudie
  mCurrentTest=get(paste0("testuser",vSimulation[2])) 
  
  vResult=NULL
  #Zeilen iterrieren pro Testdatensatz
  for(iRow in 1:nrow(mCurrentTest)){
    
    #Active User Vector auswählen
    vCurrUser=mCurrentTest[iRow,]
    
    #Testdatensatz für Triplet-Input (UserID, ItemID, Rating) umwandeln
    #Aktuelle UserID pro Zeile auflisten
    library(reshape)
    md <- melt(vCurrUser, id=(c("UserID"))) 
    #Zweite Spalte entfernen - nicht benötigt
    ActiveU <- md[,-2] #Spalte
    #Spalten umbenennen
    colnames(ActiveU) <- c("UserID", "ItemID") 
    #NA-Ratings hinzufügen
    ActiveU$Rating <- c("NA")
    
    #Write Test-Data
    write.csv(ActiveU, "Data/testdata.csv")
    
    #TF-Model aufrufen / Test
    TFTest <- system("python DNNRTest.py", wait = TRUE)
    
    # Predict-Array aus TF-Model aufrufen
    RArray <- read.table("/Result.txt", sep = ",", row.names = NULL)
    
    #Rank relevantes Item ermitteln
    vRankingList <- rank(-RArray)
    iRank <- vRankingList[2]
    
    # Speichern des Simulationsergebnisses f?r den aktiven User (Rang des relevanten Items) in Ergebnisvektor
    vResult=c(vResult,iRank)
  }
  
  # Speichern der Ergebnisvektoren in einer Matrix
  names(vResult)=NULL
  vResult=c(vSimulation,vResult)
  mResult=rbind(mResult,vResult)
  
}
