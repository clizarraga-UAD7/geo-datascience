CDF      
      	date_time         	string256         string64   @   string32       string16      string8       string4       string2       n_prof        n_param       n_levels   N   	n_history          n_calib          1   title         Argo float vertical profile    institution       CORIOLIS   source        
Argo float     history       32018-06-09T02:33:10Z csun convAGDAC.f90 Version 1.0    
references        http://www.nodc.noaa.gov/argo/     user_manual_version       3.1    Conventions       GADR-3.0 Argo-3.0 CF-1.6   featureType       trajectoryProfile      uuid      $1e77a126-5eb8-4e11-acf7-3e774161dfa9   summary       ?The U.S. National Oceanographic Data Center (NODC) operates the Argo Global Data Repository (GADR). For information about organizations contributing data to GADR, see http://www.nodc.noaa.gov/argo/      file_source       TThe Argo Global Data Assembly Center FTP server at ftp://ftp.ifremer.fr/ifremer/argo   keywords      @temperature, salinity, sea_water_temperature, sea_water_salinity   keywords_vocabulary       "NODC Data Types, CF Standard Names     creator_name      Charles Sun    creator_url       http://www.nodc.noaa.gov   creator_email         Charles.Sun@noaa.gov   id        0042682    naming_authority      gov.noaa.nodc      standard_name_vocabulary      CF-1.6     Metadata_Conventions      Unidata Dataset Discovery v1.0     publisher_name        :US DOC; NESDIS; NATIONAL OCEANOGRAPHIC DATA CENTER - IN295     publisher_url         http://www.nodc.noaa.gov/      publisher_email       NODC.Services@noaa.gov     date_created      2018-06-09T02:33:10Z   date_modified         2018-06-09T02:33:10Z   date_issued       2018-06-09T02:33:10Z   acknowledgment        }These data were acquired from the US NOAA National Oceanographic Data Center (NODC) on [DATE] from http://www.nodc.noaa.gov/.      license       ?These data are openly available to the public Please acknowledge the use of these data with the text given in the acknowledgment attribute.    cdm_data_type         trajectoryProfile      geospatial_lat_min        B<?5   geospatial_lat_max        B<?5   geospatial_lon_min        ???   geospatial_lon_max        ???   geospatial_vertical_min       @?     geospatial_vertical_max       D??    geospatial_lat_units      degrees_north      geospatial_lat_resolution         point      geospatial_lon_units      degrees_east   geospatial_lon_resolution         point      geospatial_vertical_units         decibars   geospatial_vertical_resolution        point      geospatial_vertical_positive      down   time_coverage_start       2012-07-13T22:33:06Z   time_coverage_end         2012-07-13T22:33:06Z   time_coverage_duration        point      time_coverage_resolution      point      gadr_ConventionVersion        GADR-3.0   gadr_program      convAGDAC.f90      gadr_programVersion       1.0       A   	data_type                  	long_name         	Data type      conventions       Argo reference table 1     
_FillValue                    A    format_version                 	long_name         File format version    
_FillValue                    A0   handbook_version               	long_name         Data handbook version      
_FillValue                    A4   reference_date_time                 	long_name         !Date of reference for Julian days      conventions       YYYYMMDDHHMISS     
_FillValue                    A8   date_creation                   	long_name         Date of file creation      conventions       YYYYMMDDHHMISS     
_FillValue                    AH   date_update                 	long_name         Date of update of this file    conventions       YYYYMMDDHHMISS     
_FillValue                    AX   platform_number                   	long_name         Float unique identifier    conventions       WMO float identifier : A9IIIII     
_FillValue                    Ah   project_name                  	long_name         Name of the project    
_FillValue                  @  Ap   pi_name                   	long_name         "Name of the principal investigator     
_FillValue                  @  A?   station_parameters           	            	long_name         ,List of available parameters for the station   conventions       Argo reference table 3     
_FillValue                  0  A?   cycle_number               	long_name         Float cycle number     conventions       =0...N, 0 : launch cycle (if exists), 1 : first complete cycle      
_FillValue         ??        B    	direction                  	long_name         !Direction of the station profiles      conventions       -A: ascending profiles, D: descending profiles      
_FillValue                    B$   data_centre                   	long_name         .Data centre in charge of float data processing     conventions       Argo reference table 4     
_FillValue                    B(   dc_reference                  	long_name         (Station unique identifier in data centre   conventions       Data centre convention     
_FillValue                     B,   data_state_indicator                  	long_name         1Degree of processing the data have passed through      conventions       Argo reference table 6     
_FillValue                    BL   	data_mode                  	long_name         Delayed mode or real time data     conventions       >R : real time; D : delayed mode; A : real time with adjustment     
_FillValue                    BP   platform_type                     	long_name         Type of float      conventions       Argo reference table 23    
_FillValue                     BT   float_serial_no                   	long_name         Serial number of the float     
_FillValue                     Bt   firmware_version                  	long_name         Instrument firmware version    
_FillValue                     B?   wmo_inst_type                     	long_name         Coded instrument type      conventions       Argo reference table 8     
_FillValue                    B?   juld               	long_name         ?Julian day (UTC) of the station relative to REFERENCE_DATE_TIME    standard_name         time   units         "days since 1950-01-01 00:00:00 UTC     conventions       8Relative julian days with decimal part (as parts of day)   
resolution        >?EȠ?Q)   
_FillValue        A.?~       axis      T           B?   juld_qc                	long_name         Quality on date and time   conventions       Argo reference table 2     
_FillValue                    B?   juld_location                  	long_name         @Julian day (UTC) of the location relative to REFERENCE_DATE_TIME   units         "days since 1950-01-01 00:00:00 UTC     conventions       8Relative julian days with decimal part (as parts of day)   
resolution        >??	4E?   
_FillValue        A.?~            B?   latitude               	long_name         &Latitude of the station, best estimate     standard_name         latitude   units         degree_north   
_FillValue        @?i?       	valid_min         ?V?        	valid_max         @V?        axis      Y           B?   	longitude                  	long_name         'Longitude of the station, best estimate    standard_name         	longitude      units         degree_east    
_FillValue        @?i?       	valid_min         ?f?        	valid_max         @f?        axis      X           B?   position_qc                	long_name         ,Quality on position (latitude and longitude)   conventions       Argo reference table 2     
_FillValue                    B?   positioning_system                    	long_name         Positioning system     
_FillValue                    B?   profile_pres_qc                	long_name         #Global quality flag of PRES profile    conventions       Argo reference table 2a    
_FillValue                    B?   profile_temp_qc                	long_name         #Global quality flag of TEMP profile    conventions       Argo reference table 2a    
_FillValue                    B?   profile_psal_qc                	long_name         #Global quality flag of PSAL profile    conventions       Argo reference table 2a    
_FillValue                    B?   vertical_sampling_scheme                  	long_name         Vertical sampling scheme   conventions       Argo reference table 16    
_FillValue                    B?   config_mission_number                  	long_name         :Unique number denoting the missions performed by the float     conventions       !1...N, 1 : first complete mission      
_FillValue         ??        C?   pres         
      
   	long_name         )Sea water pressure, equals 0 at sea-level      standard_name         sea_water_pressure     
_FillValue        G?O?   units         decibar    	valid_min                	valid_max         F;?    C_format      %7.1f      FORTRAN_format        F7.1   
resolution        =???   axis      Z        8  C?   pres_qc          
         	long_name         quality flag   conventions       Argo reference table 2     
_FillValue                  P  E0   pres_adjusted            
      
   	long_name         )Sea water pressure, equals 0 at sea-level      standard_name         sea_water_pressure     
_FillValue        G?O?   units         decibar    	valid_min                	valid_max         F;?    C_format      %7.1f      FORTRAN_format        F7.1   
resolution        =???   axis      Z        8  E?   pres_adjusted_qc         
         	long_name         quality flag   conventions       Argo reference table 2     
_FillValue                  P  F?   pres_adjusted_error          
         	long_name         VContains the error on the adjusted values as determined by the delayed mode QC process     
_FillValue        G?O?   units         decibar    C_format      %7.1f      FORTRAN_format        F7.1   
resolution        =???     8  G   temp         
      	   	long_name         $Sea temperature in-situ ITS-90 scale   standard_name         sea_water_temperature      
_FillValue        G?O?   units         degree_Celsius     	valid_min         ?      	valid_max         B      C_format      %9.3f      FORTRAN_format        F9.3   
resolution        :?o     8  H@   temp_qc          
         	long_name         quality flag   conventions       Argo reference table 2     
_FillValue                  P  Ix   temp_adjusted            
      	   	long_name         $Sea temperature in-situ ITS-90 scale   standard_name         sea_water_temperature      
_FillValue        G?O?   units         degree_Celsius     	valid_min         ?      	valid_max         B      C_format      %9.3f      FORTRAN_format        F9.3   
resolution        :?o     8  I?   temp_adjusted_qc         
         	long_name         quality flag   conventions       Argo reference table 2     
_FillValue                  P  K    temp_adjusted_error          
         	long_name         VContains the error on the adjusted values as determined by the delayed mode QC process     
_FillValue        G?O?   units         degree_Celsius     C_format      %9.3f      FORTRAN_format        F9.3   
resolution        :?o     8  KP   psal         
      	   	long_name         Practical salinity     standard_name         sea_water_salinity     
_FillValue        G?O?   units         psu    	valid_min         @      	valid_max         B$     C_format      %9.3f      FORTRAN_format        F9.3   
resolution        :?o     8  L?   psal_qc          
         	long_name         quality flag   conventions       Argo reference table 2     
_FillValue                  P  M?   psal_adjusted            
      	   	long_name         Practical salinity     standard_name         sea_water_salinity     
_FillValue        G?O?   units         psu    	valid_min         @      	valid_max         B$     C_format      %9.3f      FORTRAN_format        F9.3   
resolution        :?o     8  N   psal_adjusted_qc         
         	long_name         quality flag   conventions       Argo reference table 2     
_FillValue                  P  OH   psal_adjusted_error          
         	long_name         VContains the error on the adjusted values as determined by the delayed mode QC process     
_FillValue        G?O?   units         psu    C_format      %9.3f      FORTRAN_format        F9.3   
resolution        :?o     8  O?   history_institution                      	long_name         "Institution which performed action     conventions       Argo reference table 4     
_FillValue                    Z0   history_step                     	long_name         Step in data processing    conventions       Argo reference table 12    
_FillValue                    Z4   history_software                     	long_name         'Name of software which performed action    conventions       Institution dependent      
_FillValue                    Z8   history_software_release                     	long_name         2Version/release of software which performed action     conventions       Institution dependent      
_FillValue                    Z<   history_reference                        	long_name         Reference of database      conventions       Institution dependent      
_FillValue                  @  Z@   history_date                      	long_name         #Date the history record was created    conventions       YYYYMMDDHHMISS     
_FillValue                    Z?   history_action                       	long_name         Action performed on data   conventions       Argo reference table 7     
_FillValue                    Z?   history_parameter                        	long_name         (Station parameter action is performed on   conventions       Argo reference table 3     
_FillValue                    Z?   history_start_pres                    	long_name          Start pressure action applied on   
_FillValue        G?O?   units         decibar         Z?   history_stop_pres                     	long_name         Stop pressure action applied on    
_FillValue        G?O?   units         decibar         Z?   history_previous_value                    	long_name         +Parameter/Flag previous value before action    
_FillValue        G?O?        Z?   history_qctest                       	long_name         <Documentation of tests performed, tests failed (in hex form)   conventions       EWrite tests performed when ACTION=QCP$; tests failed when ACTION=QCF$      
_FillValue                    Z?   	parameter               	            	long_name         /List of parameters with calibration information    conventions       Argo reference table 3     
_FillValue                  0  P?   scientific_calib_equation               	            	long_name         'Calibration equation for this parameter    
_FillValue                    Q    scientific_calib_coefficient            	            	long_name         *Calibration coefficients for this equation     
_FillValue                    T    scientific_calib_comment            	            	long_name         .Comment applying to this parameter calibration     
_FillValue                    W    scientific_calib_date               	             	long_name         Date of calibration    conventions       YYYYMMDDHHMISS     
_FillValue                  ,  Z    crs              	long_name         Coordinate Reference System    grid_mapping_name         latitude_longitude     	epsg_code         	EPSG:4326      longitude_of_prime_meridian       0.0f   semi_major_axis       	6378137.0      inverse_flattening        298.257223563           Z,Argo profile    3.1 1.2 19500101000000  20130116114107  20180108143246  4901412 BSH                                                             Holger GIESE                                                    PRES            TEMP            PSAL               A   IF  28400973                        2C  D   APEX                            6044                            061609                          846 @?M?#Eg?1   @?M?Tò?@G?Ƨ?Cs333301   ARGOS   A   A   A   Primary sampling: discrete []                                                                                                                                                                                                                                      @?  A&ffAvffA?  A?ffA?33B??B!33B4ffBI33Bp??B???B???B?  B?  B?ffB???C?CL?C  C?3C(?fC4ffC>L?CH? CRL?C\ffCf  Co?fCz33C?  C?  C?&fC??3C?&fC??fC?&fC?ٚC?&fC??C???C??Cԙ?C?&fC???D	? D?3D??D9?D"?3D(??D/  D;??DH  DTffD`?3Dm?3Dz3D?9?D??fD???D? D?<?D?s3D?c3D?fD?C3D?p D?ɚD?	?D?P DԆfDڳ3D๚D?9?D퉚D??3D?? 111111111111111111111111111111111111111111111111111111111111111111111111111111  @ٙ?A+33A{33A?ffA???A???B  B"ffB5??BJffBr  B?34B?34B???Bș?B?  B?fgCfgC??CL?C   C)33C4?3C>??CH??CR??C\?3CfL?Cp33Cz? C?&fC?&fC?L?C??C?L?C??C?L?C?  C?L?C?33C?? C?33C?? C?L?C??3D	?3D?fD?DL?D"?fD(??D/3D;? DH33DTy?DafDm?fDz&fD?C4D?? D??gD??D?FgD?|?D?l?D? D?L?D?y?D??4D?4D?Y?DԐ Dڼ?D??4D?C4D??4D???D???111111111111111111111111111111111111111111111111111111111111111111111111111111  @??@??@??@??@??@??@??@??@??@??@??@??@??@??@??@??@??@??@??@??@??@??@??@??@??@??@??@??@??@??@??@??@??@??@??@??@??@??@??@??@??@??@??@??@??@??@??@??@??@??@??@??@??@??@??@??@??@??@??@??@??@??@??@??@??@??@??@??@??@??@??@??@??@??@??@??@??@??A???A???A???A?^5A?A??A?M?A?ȴA?`BA?1'A?-A~9XA|ZA{oAyAv??Am??An??Ao7LAm+Am&?Ai7LAh??Agl?AfVAd??Acp?A`ȴA^??A]"?A[K?AY?FAW?TAVI?AT?AQx?AM+AI?^AH?\AH-AH{AEƨA???A??A>v?A0I?A+?TA!??Az?AVAVAn?@?{@?S?@???@?
=@?S?@?(?@?%@?A?@?A?@?I?@?  @???@?b@?X@??m@???@?V@?{@~??@|Z@w??@v?+@t??@r??@p?`@o??111111111111111111111111111111111111111111111111111111111111111111111111111111  A???A???A???A?^5A?A??A?M?A?ȴA?`BA?1'A?-A~9XA|ZA{oAyAv??Am??An??Ao7LAm+Am&?Ai7LAh??Agl?AfVAd??Acp?A`ȴA^??A]"?A[K?AY?FAW?TAVI?AT?AQx?AM+AI?^AH?\AH-AH{AEƨA???A??A>v?A0I?A+?TA!??Az?AVAVAn?@?{@?S?@???@?
=@?S?@?(?@?%@?A?@?A?@?I?@?  @???@?b@?X@??m@???@?V@?{@~??@|Z@w??@v?+@t??@r??@p?`@o??111111111111111111111111111111111111111111111111111111111111111111111111111111  ;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;o;oB?VB?7B?^B?)B
=B'?BVB?\B??B??B??B?BƨB?wB?!B?B?BO?Bz?Bk?B?B_;Bs?Bm?BhsBVBD?B'?BoBB??B?mB??BŢB?B?hBaHB;dB8RB8RB8RB?B??B??B??Bs?B[#B??B??BffBffB5?B\B??B?3B?sB?5B?
B?BB?5BƨB??B?LB?9B?3B?B?B?B?-B??B?B?B??B??B??B??B??B?111111111111111111111111111111111111111111111111111111111111111111111111111111  B?FB?'B?PB?B
1B'?BU?B?VB??B?iBʡB??BƈB?WB?B?B?BO?Bz?Bk|B?B_1Bs?Bm?BhiBU?BD?B'?BaB?B??B?\B??BŐB??B?SBa1B;LB8:B8:B8:BtB??B̱B??Bs?B[B?iB?BfBBfBB5B7B?rB?B?MB?B??B?B?BƁB?\B?%B?B?B??B??B??B?B?B??B??B??B??B??B??B??B??111111111111111111111111111111111111111111111111111111111111111111111111111111  <#?
<#?
<#?
<#?
<#?
<#?
<#?
<#?
<#?
<#?
<#?
<#?
<#?
<#?
<#?
<#?
<#?
<#?
<#?
<#?
<#?
<#?
<#?
<#?
<#?
<#?
<#?
<#?
<#?
<#?
<#?
<#?
<#?
<#?
<#?
<#?
<#?
<#?
<#?
<#?
<#?
<#?
<#?
<#?
<#?
<#?
<#?
<#?
<#?
<#?
<#?
<#?
<#?
<#?
<#?
<#?
<#?
<#?
<#?
<#?
<#?
<#?
<#?
<#?
<#?
<#?
<#?
<#?
<#?
<#?
<#?
<#?
<#?
<#?
<#?
<#?
<#?
<#?
PRES            TEMP            PSAL            PRES_ADJUSTED (cycle i) = PRES (cycle i) - Surface Pressure (cycle i+1)                                                                                                                                                                                         TEMP_ADJUSTED = TEMP                                                                                                                                                                                                                                            PSAL_ADJUSTED = PSAL (re-calculated by using PRES_ADJUSTED)                                                                                                                                                                                                     Surface pressure = -0.3 dbar                                                                                                                                                                                                                                    none                                                                                                                                                                                                                                                            none                                                                                                                                                                                                                                                            Pressure adjusted by using pressure offset at the sea surface. Calibration error is manufacturer specified accuracy in dbar                                                                                                                                     No significant temperature drift detected. Calibration error is manufacturer specified accuracy with respect to ITS-90                                                                                                                                          No significant salinity drift detected (salinity adjusted for pressure offset). OW method (weighted least squares fit) adopted. The quoted error is max[0.01, 1xOW uncertainty] in PSS-78.                                                                      201411110945552014111109455520141111094555  ?  IF  ARGQCOAR1.0                                                                 20130116115048  QCP$                G?O?G?O?G?O?DEBFC           IF  ARGQCOAR1.0                                                                 20130116115048  QCF$                G?O?G?O?G?O?00000           IF      SCOO1.4                                                                 20130117150930  QC                  G?O?G?O?G?O?                GE  ARSQOW  1.0 ARGO CTD ref. database: CTD_for_DMQC_2012V01 + ARGO climatology 20130304162938  IP  PSAL            @?  D?? G?O?                GE  ARSQOW  1.0 ARGO CTD ref. database: CTD_for_DMQC_2013V01 + ARGO climatology 20141111094555  IP  PSAL            @?  D?? G?O?                IF      COFC3.0                                                                 20180108143246                      G?O?G?O?G?O?                