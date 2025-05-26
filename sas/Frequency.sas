libname project '/home/u64207882/Proiect/Input/Sas';

data ld;
   set project.ld_merge;
run;

title "Dataset for Frequency Analysis";
proc print data=ld(obs=5);
run;

/* freq and p_dist for processor family */
proc freq data=ld order=freq;
  tables Familie_procesor / nocum plots=freqplot(type=bar scale=percent) maxlevels=30;
  title "Distribution of Processor Families";
run;

proc sgplot data=ld;
   vbox price / category=Familie_procesor;
   title "Price Distribution by Processor Family";
   xaxis label="Processor Family";
   yaxis label="Price";
run;

/* freq and p_dist for operating system */
proc freq data=ld order=freq;
  tables Sistem_operare / nocum plots=freqplot(type=bar scale=percent) maxlevels=15;
  title "Distribution of Operating Systems";
run;

proc sgplot data=ld;
   vbox price / category=Sistem_operare;
   title "Price Distribution by Operating System";
   xaxis label="Operating System";
   yaxis label="Price";
run;


/* freq and avg price for touchscreen feature */
proc freq data=ld order=freq;
  tables TouchScreen / nocum plots=freqplot(type=bar scale=percent);
  title "Distribution of Laptops with Touchscreens";
run;
proc means data=ld mean median;
   class TouchScreen;
   var price;
   title "Average Price by Touchscreen Feature";
run;


/* freq and avg price for backlit keyboard*/
proc freq data=ld order=freq;
  tables Tastatura_iluminata / nocum plots=freqplot(type=bar scale=percent);
  title "Distribution of Laptops with Backlit Keyboards";
run;

proc means data=ld mean median;
   class Tastatura_iluminata;
   var price;
   title "Average Price by Backlit Keyboard Feature";
run;

/* ====== Price rels ==== */
/* Histogram of Laptop Prices */
proc sgplot data=ld;
  histogram price / binwidth=2000; /* Adjust binwidth as needed */
  title "Distribution of Laptop Prices";
  xaxis label="Price" MAX=20000;
run;


/* Price buckets */
data ld;
    set ld;
    length Price_Category $15.;
    if missing(price) then Price_Category = 'Unknown';
    else if price <= 3000 then Price_Category = 'Budget';
    else if price <= 6000 then Price_Category = 'Mid-Range';
    else Price_Category = 'Premium';
run;

proc freq data=ld_merge_enhanced;
    tables Price_Category;
    title "Distribution of Laptops by Price Category";
run;

/* price vs ram */ 
proc sgplot data=ld;
    title "Price vs. RAM (Bubble Size by SSD Capacity)";
    bubble x=Capacitate_RAM_MB y=price size=Capacitate_RAM_MB / group=manufacturer;
    xaxis label="RAM (MB)";
    yaxis label="Price";
    keylegend / location=outside position=bottom;
run;

/* Example: Average Price by Manufacturer and Price Category */
proc means data=ld_merge_enhanced noprint;
    class manufacturer Price_Category;
    var price;
    output out=avg_price_mfg_cat mean=AvgPrice;
run;


proc sgplot data=avg_price_mfg_cat;
    title "Average Price by Manufacturer and Price Category";
    vbar Price_Category / response=AvgPrice group=manufacturer groupdisplay=cluster;
    xaxis label="Price Category";
    yaxis label="Average Price";
    keylegend /
    title="Manufacturer";
run;

/* Histogram ssd size */
proc sgplot data=ld_merge;
  histogram Capacitate_SSD_GB;
  title "Distribution of SSD Capacity";
  xaxis label="SSD (GB)";
run;

libname myjson clear;
filename mydata clear;
title; 
