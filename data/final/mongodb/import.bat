@echo off
REM MongoDB Import Script for Windows
SET DB_NAME=pricecheck_tn
echo Starting MongoDB import...

echo Importing users...
mongoimport --db %DB_NAME% --collection users --file data\final\mongodb\users.json --jsonArray

echo Importing user_profiles...
mongoimport --db %DB_NAME% --collection user_profiles --file data\final\mongodb\user_profiles.json --jsonArray

echo Importing categories...
mongoimport --db %DB_NAME% --collection categories --file data\final\mongodb\categories.json --jsonArray

echo Importing stores...
mongoimport --db %DB_NAME% --collection stores --file data\final\mongodb\stores.json --jsonArray

echo Importing products...
mongoimport --db %DB_NAME% --collection products --file data\final\mongodb\products.json --jsonArray

echo Importing reviews...
mongoimport --db %DB_NAME% --collection reviews --file data\final\mongodb\reviews.json --jsonArray

echo Importing price_history...
mongoimport --db %DB_NAME% --collection price_history --file data\final\mongodb\price_history.json --jsonArray

echo Importing search_history...
mongoimport --db %DB_NAME% --collection search_history --file data\final\mongodb\search_history.json --jsonArray

echo Import complete!
pause