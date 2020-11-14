# Project 2 - Ames Iowa Housing Price Prediction and Kaggle Challenge

## Problem Statement

The Ames Housing Dataset was introduced by Professor Dean De Cock in 2011. It contains 2051 observations of housing sales in Ames, Iowa between 2006 and 2010. There are 23 nominal, 23 ordinal, 14 discrete, and 21 continuous variables describing each house’s size, quality, area, age, and other miscellaneous attributes.

Real estate is a tangible asset made up of property and the land on which it sits. Housing price ranges are of great interest for both buyers and sellers. People looking to buy a new home tend to be more conservative with their budgets and market strategies. 

Therefore, it is important to investigate:
1. What are the features that will value-add a home and increase its sales price
2. The predicted sales price of a home if a combination of features is given
3. Likewise, predict feature combinations based on a given budget

The above questions will be interested by audience such as home buyers, home owners planning/not planning to sell their homes and property agents.

In this project, I will use Ames housing data to create linear regression models that predicts the price of houses in Ames, Iowa.

The types of models that will be created are linear Regression, ridge, Lasso and Elastic Net. The best model will be selected for housing price prediction investigations mentioned.

The success of the model will be determined by the root mean squared error (RMSE) value. It is a measure of how accurately the model predicts the sales price. Lower values of RMSE indicate better fit.

This project will demostrates:
1. Data Inspection & Cleaning
2. Exploratory Data Analysis
3. Preprocessing
4. Feature Engineering
5. Modeling and Regularization Interpretation
6. Recommendations of value-added features based on research and modeling results


## Executive Summary

The purpose of this project is to select an appropriate model to predict sales price of homes in Ames Iowa. This model can benefit 3 groups of people, namely, home buyers who want to buy a value-for-money ideal home, home sellers who wanted to maximise profit when selling a house and property agents who wanted to close transactions efficiently. 

The model selected was Lasso model, it identified homes can be sold at higher price if they were bigger, better overall quality which includes heating (this is important as Ames experience severe winters), exterior, basement, fireplace and kitchen quality. Popular homes comes with more fireplaces, a masonry veneer area, poured concrete foundation, more rooms, more baths and bigger garage.

Homes that can sell at high prices is located at Northridge heights, which is described to be of low crime rate, near to a leisure park, 4 nearby schools, population growth rate of 523% and is suitable for retirement.

Homes that are brand new increases sales price. Likewise, price drop as age of the home increases.

The above predictions are important to home buyers, owners and property agents. By knowing what features is strongly linear correlated to sales price, they can make necessary improvement. Example, home owners can renovate, repaint, fix broken heating system, kitchens, basement to ensure the quality of homes for higher price. Home buyers can pioritize features of interest to keep within their budget. Home buyers can also ensure they do not overpay for a home by applying the model. Property agents on the other hand, can use the model to show potential clients whether that the house is worth buying and is it at a resonable price. This can help property agents to close deals more efficiently.

In this project, we are able to find out what are the features value-add a home and increase its sales price. This was addressed earlier at the top of this section. On top of this, we can predict the sales price based on a given set of features. Next, we can predict feature combinations based on a given budget. This can be done as the model generates a coefficent that can tell us how much sales price increase with a given unit change of the feature measurement.

Nevertheless, our predictions still have limitations:
- Sales price is highly impacted by economic climate, salary, occupation, employment rate, population demographics etc. Wider feature types is needed to be collected for more accurate real life prediction.
- Dataset scope is from 2006 to 2010, the prediction will be outdated for real life application. More recent dataset will be needed for current predictions.
- Further fine tuning eg data cleaning, reduce multi-collinearity, feature engineering and feature selection can be done to improve prediction capability.
- Prediction is applicable to Ames housing neighbourhood only. Not applicable to other neighbourhoods beyond Ames housing.


## Data Dictionary
| Feature 	| Type 	| Description 	| Values 	|
|-	|-	|-	|-	|
| saleprice 	| int 	| The property's sale price in dollars (Target Value) 	| USD 	|
| id 	| int 	| ID value for sale row 	| integer 	|
| pid 	| int 	| ID value for property 	| integer 	|
| ms_subclass 	| int 	| The building class 	| 20 - 1-STORY 1946 & NEWER ALL STYLES<br>30 - 1-STORY 1945 & OLDER<br>40 - 1-STORY W/FINISHED ATTIC ALL AGES<br>45 - 1-1/2 STORY - UNFINISHED ALL AGES<br>50 - 1-1/2 STORY FINISHED ALL AGES<br>60 - 2-STORY 1946 & NEWER<br>70 - 2-STORY 1945 & OLDER<br>75 - 2-1/2 STORY ALL AGES<br>80 - SPLIT OR MULTI-LEVEL<br>85 - SPLIT FOYER<br>90 - DUPLEX - ALL STYLES AND AGES<br>120 - 1-STORY PUD (Planned Unit Development) - 1946 & NEWER<br>150 - 1-1/2 STORY PUD - ALL AGES<br>160 - 2-STORY PUD - 1946 & NEWER<br>180 - PUD - MULTILEVEL - INCL SPLIT LEV/FOYER<br>190 - 2 FAMILY CONVERSION - ALL STYLES AND AGES 	|
| ms_zoning 	| object 	| Identifies the general zoning classification of the sale 	| A - Agriculture<br>C - Commercial<br>FV - Floating Village Residential<br>I - Industrial<br>RH - Residential High Density<br>RL - Residential Low Density<br>RP - Residential Low Density Park<br>RM - Residential Medium Density 	|
| bldg_type 	| object 	| Type of dwelling 	| 1Fam - Single-family Detached<br>2FmCon - Two-family Conversion; originally built as one-family dwelling<br>Duplx - Duplex<br>TwnhsE - Townhouse End Unit<br>TwnhsI - Townhouse Inside Unit 	|
| house_style 	| object 	| Style of dwelling 	| 1Story - One story<br>1.5Fin - One and one-half story: 2nd level finished<br>1.5Unf - One and one-half story: 2nd level unfinished<br>2Story - Two story<br>2.5Fin - Two and one-half story: 2nd level finished<br>2.5Unf - Two and one-half story: 2nd level unfinished<br>SFoyer - Split Foyer<br>SLvl - Split Level 	|
| lot_frontage 	| float 	| Linear feet of street connected to property 	| ft 	|
| lot_area 	| int 	| Lot size in square feet 	| sq ft 	|
| street 	| int 	| Type of road access to property 	| Grvl - Gravel<br>Pave - Paved 	|
| alley 	| object 	| Type of alley access to property 	| Grvl - Gravel<br>Pave - Paved<br>NA - No alley access 	|
| lot_shape 	| int 	| General shape of property 	| Reg - Regular<br>IR1 - Slightly irregular<br>IR2 - Moderately Irregular<br>IR3 - Irregular 	|
| land_contour 	| int 	| Flatness of the property 	| Lvl - Near Flat/Level<br>Bnk - Banked - Quick and significant rise from street grade to building<br>HLS - Hillside - Significant slope from side to side<br>Low - Depression 	|
| utilities 	| int 	| Type of utilities available 	| AllPub - All public Utilities (E,G,W,& S)<br>NoSewr - Electricity, Gas, and Water (Septic Tank)<br>NoSeWa - Electricity and Gas Only<br>ELO - Electricity only 	|
| lot_config 	| object 	| Lot configuration 	| Inside - Inside lot<br>Corner - Corner lot<br>CulDSac - Cul-de-sac<br>FR2 - Frontage on 2 sides of property<br>FR3 - Frontage on 3 sides of property 	|
| land_slope 	| int 	| Slope of property 	| Gtl - Gentle slope<br>Mod - Moderate Slope<br>Sev - Severe Slope 	|
| neighborhood 	| object 	| Physical locations within Ames city limits 	| Blmngtn - Bloomington Heights<br>Blueste - Bluestem<br>BrDale - Briardale<br>BrkSide - Brookside<br>ClearCr - Clear Creek<br>CollgCr - College Creek<br>Crawfor - Crawford<br>Edwards - Edwards<br>Gilbert - Gilbert<br>IDOTRR - Iowa DOT and Rail Road<br>MeadowV - Meadow Village<br>Mitchel - Mitchell<br>Names - North Ames<br>NoRidge - Northridge<br>NPkVill - Northpark Villa<br>NridgHt - Northridge Heights<br>NWAmes - Northwest Ames<br>OldTown - Old Town<br>SWISU - South & West of Iowa State University<br>Sawyer - Sawyer<br>SawyerW - Sawyer West<br>Somerst - Somerset<br>StoneBr - Stone Brook<br>Timber - Timberland<br>Veenker - Veenker 	|
| condition_1 	| object 	| Proximity to main road or railroad 	| Artery - Adjacent to arterial street<br>Feedr - Adjacent to feeder street<br>Norm - Normal<br>RRNn - Within 200' of North-South Railroad<br>RRAn - Adjacent to North-South Railroad<br>PosN - Near positive off-site feature--park, greenbelt, etc.<br>PosA - Adjacent to postive off-site feature<br>RRNe - Within 200' of East-West Railroad<br>RRAe - Adjacent to East-West Railroad 	|
| condition_2 	| object 	| Proximity to main road or railroad (if a second is present)  	| Artery - Adjacent to arterial street<br>Feedr - Adjacent to feeder street<br>Norm - Normal<br>RRNn - Within 200' of North-South Railroad<br>RRAn - Adjacent to North-South Railroad<br>PosN - Near positive off-site feature--park, greenbelt, etc.<br>PosA - Adjacent to postive off-site feature<br>RRNe - Within 200' of East-West Railroad<br>RRAe - Adjacent to East-West Railroad 	|
| overall_qual 	| int 	| Overall material and finish quality 	| 1 - Very Poor <-> 10 - Very Excellent 	|
| overall_cond 	| int 	| Overall condition rating 	| 1 - Very Poor <-> 10 - Very Excellent 	|
| year_built 	| int 	| Original construction date 	| year 	|
| year_remod/add 	| int 	| Remodel date (same as construction date if no remodeling or additions) 	| year 	|
| roof_style 	| object 	| Type of roof 	| Flat - Flat<br>Gable - Gable<br>Gambrel - Gabrel (Barn)<br>Hip - Hip<br>Mansard - Mansard<br>Shed - Shed 	|
| roof_matl 	| object 	| Roof material 	| ClyTile - Clay or Tile<br>CompShg - Standard (Composite) Shingle<br>Membran - Membrane<br>Metal - Metal<br>Roll - Roll<br>Tar&Grv - Gravel & Tar<br>WdShake - Wood Shakes<br>WdShngl - Wood Shingles 	|
| exterior_1st 	| object 	| Exterior covering on house 	| AsbShng - Asbestos Shingles<br>AsphShn - Asphalt Shingles<br>BrkComm - Brick Common<br>BrkFace - Brick Face<br>CBlock - Cinder Block<br>CemntBd - Cement Board<br>HdBoard - Hard Board<br>ImStucc - Imitation Stucco<br>MetalSd - Metal Siding<br>Other - Other<br>Plywood - Plywood<br>PreCast - PreCast<br>Stone - Stone<br>Stucco - Stucco<br>VinylSd - Vinyl Siding<br>WdSdng - Wood Siding<br>WdShing - Wood Shingles 	|
| exterior_2nd 	| object 	| Exterior covering on house (if more than one material) 	| AsbShng - Asbestos Shingles<br>AsphShn - Asphalt Shingles<br>BrkComm - Brick Common<br>BrkFace - Brick Face<br>CBlock - Cinder Block<br>CemntBd - Cement Board<br>HdBoard - Hard Board<br>ImStucc - Imitation Stucco<br>MetalSd - Metal Siding<br>Other - Other<br>Plywood - Plywood<br>PreCast - PreCast<br>Stone - Stone<br>Stucco - Stucco<br>VinylSd - Vinyl Siding<br>WdSdng - Wood Siding<br>WdShing - Wood Shingles 	|
| mas_vnr_type 	| object 	| Masonry veneer type 	| BrkCmn - Brick Common<br>BrkFace - Brick Face<br>CBlock - Cinder Block<br>None - None<br>Stone - Stone 	|
| mas_vnr_area 	| float 	| Masonry veneer area in square feet 	| sq ft 	|
| exter_qual 	| int 	| Exterior material quality 	| Ex - Excellent<br>Gd - Good<br>TA - Average/Typical<br>Fa - Fair<br>Po - Poor 	|
| exter_cond 	| int 	| Present condition of the material on the exterior 	| Ex - Excellent<br>Gd - Good<br>TA - Average/Typical<br>Fa - Fair<br>Po - Poor 	|
| foundation 	| object 	| Type of foundation 	| BrkTil - Brick & Tile<br>CBlock - Cinder Block<br>PConc - Poured Contrete<br>Slab - Slab<br>Stone - Stone<br>Wood - Wood 	|
| bsmt_qual 	| int 	| Height of the basement 	| Ex - Excellent (100+ inches)<br>Gd - Good (90-99 inches)<br>TA - Typical (80-89 inches)<br>Fa - Fair (70-79 inches)<br>Po - Poor (<70 inches)<br>NA - No Basement 	|
| bsmt_cond 	| object 	| General condition of the basement 	| Ex - Excellent<br>Gd - Good<br>TA - Typical - slight dampness allowed<br>Fa - Fair - dampness or some cracking or settling<br>Po - Poor - Severe cracking, settling, or wetness<br>NA - No Basement 	|
| bsmt_exposure 	| int 	| Walkout or garden level basement walls 	| Gd - Good Exposure<br>Av - Average Exposure (split levels or foyers typically score average or above)<br>Mn - Mimimum Exposure<br>No - No Exposure<br>NA - No Basement 	|
| bsmtfin_type_1 	| object 	| Quality of basement finished area 	| GLQ - Good Living Quarters<br>ALQ - Average Living Quarters<br>BLQ - Below Average Living Quarters<br>Rec - Average Rec Room<br>LwQ - Low Quality<br>Unf - Unfinshed<br>NA - No Basement 	|
| bsmtfin_sf_1 	| float 	| Type 1 finished square feet 	| sq ft 	|
| bsmtfin_type_2 	| object 	| Quality of second finished area (if present) 	| GLQ - Good Living Quarters<br>ALQ - Average Living Quarters<br>BLQ - Below Average Living Quarters<br>Rec - Average Rec Room<br>LwQ - Low Quality<br>Unf - Unfinshed<br>NA - No Basement 	|
| bsmtfin_sf_2 	| float 	| Type 2 finished square feet 	| sq ft 	|
| bsmt_unf_sf 	| float 	| Unfinished square feet of basement area 	| sq ft 	|
| total_bsmt_sf 	| float 	| Total square feet of basement area 	| sq ft 	|
| heating 	| object 	| Type of heating 	| Floor - Floor Furnace<br>GasA - Gas forced warm air furnace<br>GasW - Gas hot water or steam heat<br>Grav - Gravity furnace<br>OthW - Hot water or steam heat other than gas<br>Wall - Wall furnace 	|
| heating_qc 	| int 	| Heating quality and condition 	| Ex - Excellent<br>Gd - Good<br>TA - Average/Typical<br>Fa - Fair<br>Po - Poor 	|
| electrical 	| object 	| Electrical system 	| SBrkr - Standard Circuit Breakers & Romex<br>FuseA - Fuse Box over 60 AMP and all Romex wiring (Average)<br>FuseF - 60 AMP Fuse Box and mostly Romex wiring (Fair)<br>FuseP - 60 AMP Fuse Box and mostly knob & tube wiring (poor)<br>Mix - Mixed 	|
| central_air 	| int 	| Central air conditioning 	| Y/N 	|
| first_flr_sf 	| int 	| First Floor square feet 	| sq ft 	|
| second_fl_sf 	| int 	| Second floor square feet 	| sq ft 	|
| low_qual_fin_sf 	| int 	| Low quality finished square feet (all floors) 	| sq ft 	|
| gr_liv_area 	| int 	| Above grade (ground) living area square feet 	| sq ft 	|
| bsmt_full_bath 	| float 	| Basement full bathrooms 	| number 	|
| bsmt_half_bath 	| float 	| Basement half bathrooms 	| number 	|
| full_bath 	| int 	| Full bathrooms above grade 	| integer 	|
| half_bath 	| int 	| Half baths above grade 	| integer 	|
| bedroom_abvgr 	| int 	| Number of bedrooms above basement level 	| integer 	|
| kitchen_abvgr 	| int 	| Number of kitchens 	| integer 	|
| kitchen_qual 	| int 	| Kitchen quality 	| Ex - Excellent<br>Gd - Good<br>TA - Typical/Average<br>Fa - Fair<br>Po - Poor 	|
| totrms_abvgrd 	| int 	| Total rooms above grade (does not include bathrooms) 	| integer 	|
| functional 	| object 	| Home functionality rating 	| Typ - Typical Functionality<br>Min1 - Minor Deductions 1<br>Min2 - Minor Deductions 2<br>Mod - Moderate Deductions<br>Maj1 - Major Deductions 1<br>Maj2 - Major Deductions 2<br>Sev - Severely Damaged<br>Sal - Salvage only 	|
| fireplaces 	| int 	| Number of fireplaces 	| integer 	|
| fireplace_qu 	| int 	| Fireplace quality 	| Ex - Excellent - Exceptional Masonry Fireplace<br>Gd - Good - Masonry Fireplace in main level<br>TA - Average - Prefabricated Fireplace in main living area or Masonry Fireplace in basement<br>Fa - Fair - Prefabricated Fireplace in basement<br>Po - Poor - Ben Franklin Stove<br>NA - No Fireplace 	|
| garage_type 	| object 	| Garage location 	| 2Types - More than one type of garage<br>Attchd - Attached to home<br>Basment - Basement Garage<br>BuiltIn - Built-In (Garage part of house - typically has room above garage)<br>CarPort - Car Port<br>Detchd - Detached from home<br>NA - No Garage 	|
| garage_yr_blt 	| object 	| Year garage was built 	| year 	|
| garage_finish 	| int 	| Interior finish of the garage 	| Fin - Finished<br>RFn - Rough Finished<br>Unf - Unfinished<br>NA - No Garage 	|
| garage_cars 	| float 	| Size of garage in car capacity 	| number 	|
| garage_area 	| float 	| Size of garage in square feet 	| sq ft 	|
| garage_qual 	| int 	| Garage quality 	| Ex - Excellent<br>Gd - Good<br>TA - Typical/Average<br>Fa - Fair<br>Po - Poor<br>NA - No Garage 	|
| garage_cond 	| int 	| Garage condition 	| Ex - Excellent<br>Gd - Good<br>TA - Typical/Average<br>Fa - Fair<br>Po - Poor<br>NA - No Garage 	|
| paved_drive 	| int 	| Paved driveway 	| Y - Paved<br>P - Partial Pavement<br>N - Dirt/Gravel 	|
| wood_deck_sf 	| int 	| Wood deck area in square feet 	| sq ft 	|
| OpenPorchSF 	| int 	| Open porch area in square feet 	| sq ft 	|
| EnclosedPorch 	| int 	| Enclosed porch area in square feet 	| sq ft 	|
| ThreeSsnPorch 	| int 	| Three season porch area in square feet 	| sq ft 	|
| ScreenPorch 	| int 	| Screen porch area in square feet 	| sq ft 	|
| PoolArea 	| int 	| Pool area in square feet 	| sq ft 	|
| PoolQC 	| int 	| Pool quality 	| Ex - Excellent<br>Gd - Good<br>TA - Average/Typical<br>Fa - Fair<br>NA - No Pool 	|
| Fence 	| int 	| Fence quality 	| GdPrv - Good Privacy<br>MnPrv - Minimum Privacy<br>GdWo - Good Wood<br>MnWw - Minimum Wood/Wire<br>NA - No Fence 	|
| MiscFeature 	| object 	| Miscellaneous feature not covered in other categories 	| Elev - Elevator<br>Gar2 - 2nd Garage (if not described in garage section)<br>Othr - Other<br>Shed - Shed (over 100 SF)<br>TenC - Tennis Court<br>NA - None 	|
| MiscVal 	| int 	| Value of miscellaneous feature 	| USD 	|
| MoSold 	| int 	| Month sold 	| 1 - January <-> 12 - December 	|
| YrSold 	| int 	| Year sold 	| year 	|
| SaleType 	| object 	| Type of sale 	| WD - Warranty Deed - Conventional<br>CWD - Warranty Deed - Cash<br>VWD - Warranty Deed - VA Loan<br>New - Home just constructed and sold<br>COD - Court Officer Deed/Estate<br>Con - Contract 15% Down payment regular terms<br>ConLw - Contract Low Down payment and low interest<br>ConLI - Contract Low Interest<br>ConLD - Contract Low Down<br>Oth - Other 	|
| Age 	| int 	| Age of the house before sold 	| Year sold - Year built 	|
| RemodeledAge 	| int 	| Years remodeled before sold 	| Year remodeled - Year built 	|
| TotalLivArea 	| int 	| Total living area in the house 	| Ground living area + total basement square feet 	|
| TotalBath 	| int 	| Total number of baths in the house 	| Full baths + half baths(*2) 	|


## Conclusion
Lets recap our problem statement!

**What are the features that will value-add a home and increase its sales price?**\
Based on our top 20 prodictors, we see total living area (basement + ground area) and garage area size being valued. Features like number of fireplaces, number of rooms, number of baths, with masonary veneer area and poured concrete foundation are important to buyers too. Overall house quality, kitchen quality, exterior quality, best basement finishing will also increase sales price. Better heating quality helped to increase the price due to severe winters in Ames county.

Brand new house fetch the best price. Price dropped linearly as age increase.
In specific, need to mention Northridge heights neighbourhood is the most popular choice and reasons were stated under research section.

**The predicted sales price of a home if a combination of features is given?**\
This is achieved by Kaggle competition submission whereby a test.csv containing its features were provided to predict the sales price using selected prediction model by the contestant. The model predicted sales prices were then feed into kaggle and compared against the actual. Kaggle will return the RMSE score based on actual sales price. My RMSE score is 33067. Further fine tuning eg data cleaning, reduce multi-collinearity, feature engineering and feature selection can be done to improve the scoring.

Model selected was lasso model. Standard deviation is low ~0.013. R2 is around ~0.86. Generally, all 4 models created perform very similarly. Lasso was chosen because it gives the best RMSE score on kaggle contest among the 4 models. Observed that elastic net has l1_ratio of 0.94. This shows lasso is dominant to suppress overfitting than Ridge.

**Predict feature combinations based on a given budget?**\
With the given budget, one can look up Top 20 predictors based on Lasso model. We can work backwards since we have the budget y and coefficient, to obtain various X features measurement in combinations. Example, the coefficient of overall quality is $12960.98, which means for every increase of 1 unit in overall quality, the price of the home will increase by 12960.98. By combining those features of interest and multiply by respective coefficients, we can obtain the sales price of a home within our budget

As mentioned, in Outside Research section, sales price is highly impacted by economic climate, salary, occupation, employment rate, population demographics etc. Wider feature types is needed to be collected for more accurate real like prediction.

Dataset scope is from 2006 to 2010, the prediction will be outdated for real life application. More recent dataset will be needed for current predictions.

Prediction is limited to Ames housing neighbourhood only. As the training dataset does not include other housing neighbourhood.


## Recommendations

**1. Home buyers**\
Bigger homes, poured concrete foundation, new homes, Northridge heights neighbourhood, with a masonary veneer area, bigger garage, bigger basement, more fireplaces will be more expensive. The prediction model can help to save time during house hunting and to ensure buyer did not overpay for a house. In addition, home buyers can also use the model to work out features of interest alongside with their budget constraints.

**2. Home owners**\
It was recommended to improve the home quality for the exterior, kitchen, fireplace, by renovating if the house is of bad condition or rundown. Repair the heating as well if it is not working. Repaint the walls if needed. Maintain the home to at least functioning condition. Remodelling can also be done to increase the living area.

**3. Property agents**\
Property agent can use my model to recommend appropriate homes that is within buyer's budget. This will save time for the client during house hunting. Property agent can also use the model to present the selling/buying price to clients, so convince them that the price is reasonable. This can help to close deals at a faster rate.

