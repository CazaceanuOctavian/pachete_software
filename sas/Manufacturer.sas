libname project '/home/u64207882/Proiect/Input/Sas';

data ld;
   set project.ld_merge;
run;

proc freq data=ld order=freq;
  tables manufacturer / nocum plots=freqplot(type=bar scale=percent) maxlevels=20;
  title "Distribution of Laptops by Manufacturer";
run;

proc means data=ld mean median;
   class manufacturer;
   var price Capacitate_RAM_MB Capacitate_SSD_GB;
   output out=mfg_avg_specs mean(price)=AvgPrice mean(Capacitate_RAM_MB)=AvgRAM mean(Capacitate_SSD_GB)=AvgSSD;
   title "Average Specifications by Manufacturer";
run;

proc print data=mfg_avg_specs;
run;

proc freq data=ld;
   tables manufacturer*Price_Category / nopercent norow nocol;
   title "Manufacturer Distribution Across Price Categories";
run;
