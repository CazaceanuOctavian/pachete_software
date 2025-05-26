filename mydata "/home/u64207882/Proiect/Input/evomag_2024_11_13.json";
libname myjson json fileref=mydata;

data ld_root;
/*   set myjson.specifications;  */
  set myjson.root;
run;
data ld_spec;
  set myjson.specifications; 
run;

data ld_spec_t;
    set myjson.specifications;

    length Capacitate_SSD_GB 8; 

    if missing(Capacitate_SSD) or Capacitate_SSD = '' then do;
        Capacitate_SSD_GB = .;
    end;
    else do;
        char_numeric_part = compress(Capacitate_SSD, , 'kd');
        if char_numeric_part ne '' then do;
             Capacitate_SSD_GB = input(char_numeric_part, best12.);
        end;
        else do;
            Capacitate_SSD_GB = .;
            if not missing(Capacitate_SSD) then
                put "NOTE: Unexpected format for Capacitate_SSD: " Capacitate_SSD " - could not extract numeric part.";
        end;
    end;

    drop char_numeric_part; 
  
run;

DATA ld_merge;
    MERGE ld_root (IN=in1) 
          ld_spec_t (IN=in2);
    BY ordinal_root;
    IF in1 AND in2;
RUN;

libname mylib '/home/u64207882/Proiect/Input/Sas';
DATA mylib.ld_merge;
    SET work.ld_merge;
RUN;


/* Histogram of Laptop Prices (you have PROC MEANS, but a visual is good) */
proc sgplot data=ld_merge;
  histogram price / binwidth=2000; /* Adjust binwidth as needed */
  title "Distribution of Laptop Prices";
  xaxis label="Price" MAX=20000;
run;

/* Histogram of RAM Capacity */
proc sgplot data=ld_merge;
  histogram Numar_nuclee;
  density Numar_nuclee;
  title "Nr nuclee";
  xaxis label="Nr. Nuclee";
run;

/* Histogram of SSD Capacity */
proc sgplot data=ld_merge;
  histogram Capacitate_SSD_GB;
  title "Distribution of SSD Capacity";
  xaxis label="SSD (GB)";
run;