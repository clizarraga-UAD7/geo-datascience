CDF      
      N_PROF        N_LEVELS  �   N_CALIB       STRING2       STRING4       STRING8       STRING16      STRING32       STRING64   @   	STRING256         	DATE_TIME         N_PARAM       	N_HISTORY                title         Argo float vertical profile    institution       CSIRO      source        
Argo float     history       2022-03-24T00:31:41Z creation      
references        (http://www.argodatamgt.org/Documentation   user_manual_version       3.2    Conventions       Argo-3.2 CF-1.6    featureType       trajectoryProfile         @   	DATA_TYPE                  
_FillValue               	long_name         	Data type      conventions       Argo reference table 1          6x   FORMAT_VERSION                 
_FillValue               	long_name         File format version         6�   HANDBOOK_VERSION               
_FillValue               	long_name         Data handbook version           6�   REFERENCE_DATE_TIME       
         
_FillValue               	long_name         !Date of reference for Julian days      conventions       YYYYMMDDHHMISS          6�   DATE_CREATION         
         
_FillValue               	long_name         Date of file creation      conventions       YYYYMMDDHHMISS          6�   DATE_UPDATE       
         
_FillValue               	long_name         Date of update of this file    conventions       YYYYMMDDHHMISS          6�   PLATFORM_NUMBER                    
_FillValue               	long_name         Float unique identifier    conventions       WMO float identifier : A9IIIII          6�   PROJECT_NAME                   
_FillValue               	long_name         Name of the project       @  6�   PI_NAME                    
_FillValue               	long_name         "Name of the principal investigator        @  7   STATION_PARAMETERS                        
_FillValue               conventions       Argo reference table 3     	long_name         ,List of available parameters for the station      0  7H   CYCLE_NUMBER                
_FillValue         ��   	long_name         Float cycle number     conventions       =0...N, 0 : launch cycle (if exists), 1 : first complete cycle           7x   	DIRECTION                   
_FillValue               	long_name         !Direction of the station profiles      conventions       -A: ascending profiles, D: descending profiles           7|   DATA_CENTRE                    
_FillValue               	long_name         .Data centre in charge of float data processing     conventions       Argo reference table 4          7�   DC_REFERENCE                   
_FillValue               	long_name         (Station unique identifier in data centre   conventions       Data centre convention           7�   DATA_STATE_INDICATOR                   
_FillValue               	long_name         1Degree of processing the data have passed through      conventions       Argo reference table 6          7�   	DATA_MODE                   
_FillValue               	long_name         Delayed mode or real time data     conventions       >R : real time; D : delayed mode; A : real time with adjustment          7�   PLATFORM_TYPE                      
_FillValue               	long_name         Type of float      conventions       Argo reference table 23          7�   FLOAT_SERIAL_NO                    
_FillValue               	long_name         Serial number of the float           7�   FIRMWARE_VERSION                   
_FillValue               	long_name         Instrument firmware version          7�   WMO_INST_TYPE                      
_FillValue               	long_name         Coded instrument type      conventions       Argo reference table 8          8   JULD                
_FillValue        A.�~       standard_name         time   	long_name         ?Julian day (UTC) of the station relative to REFERENCE_DATE_TIME    conventions       8Relative julian days with decimal part (as parts of day)   units         "days since 1950-01-01 00:00:00 UTC     
resolution        >�����h�   axis      T           8   JULD_QC                 
_FillValue               	long_name         Quality on date and time   conventions       Argo reference table 2          8   JULD_LOCATION                   
_FillValue        A.�~       	long_name         @Julian day (UTC) of the location relative to REFERENCE_DATE_TIME   units         "days since 1950-01-01 00:00:00 UTC     conventions       8Relative julian days with decimal part (as parts of day)   
resolution        >�����h�   axis      T           8   LATITUDE                
_FillValue        @�i�       	long_name         &Latitude of the station, best estimate     standard_name         latitude   units         degree_north   	valid_min         �V�        	valid_max         @V�        axis      Y           8$   	LONGITUDE                   
_FillValue        @�i�       	long_name         'Longitude of the station, best estimate    standard_name         	longitude      units         degree_east    	valid_min         �f�        	valid_max         @f�        axis      X           8,   POSITION_QC                 
_FillValue               	long_name         ,Quality on position (latitude and longitude)   conventions       Argo reference table 2          84   POSITIONING_SYSTEM                     
_FillValue               	long_name         Positioning system          88   PROFILE_PRES_QC                 
_FillValue               	long_name         #Global quality flag of PRES profile    conventions       Argo reference table 2a         8@   PROFILE_TEMP_QC                 
_FillValue               	long_name         #Global quality flag of TEMP profile    conventions       Argo reference table 2a         8D   PROFILE_PSAL_QC                 
_FillValue               	long_name         #Global quality flag of PSAL profile    conventions       Argo reference table 2a         8H   VERTICAL_SAMPLING_SCHEME          	         
_FillValue               	long_name         Vertical sampling scheme   conventions       Argo reference table 16         8L   CONFIG_MISSION_NUMBER                   
_FillValue         ��   	long_name         :Unique number denoting the missions performed by the float     conventions       !1...N, 1 : first complete mission           9L   PRES                
   
_FillValue        G�O�   	long_name         )Sea water pressure, equals 0 at sea-level      standard_name         sea_water_pressure     units         decibar    	valid_min                	valid_max         F;�    C_format      %7.1f      FORTRAN_format        F7.1   
resolution        =���   axis      Z        t  9P   PRES_QC                    
_FillValue               	long_name         quality flag   conventions       Argo reference table 2       �  G�   PRES_ADJUSTED                   	   
_FillValue        G�O�   	long_name         )Sea water pressure, equals 0 at sea-level      standard_name         sea_water_pressure     units         decibar    	valid_min                	valid_max         F;�    C_format      %7.1f      FORTRAN_format        F7.1   
resolution        =���     t  Kd   PRES_ADJUSTED_QC                   
_FillValue               	long_name         quality flag   conventions       Argo reference table 2       �  Y�   PRES_ADJUSTED_ERROR                    
_FillValue        G�O�   	long_name         VContains the error on the adjusted values as determined by the delayed mode QC process     units         decibar    C_format      %7.1f      FORTRAN_format        F7.1   
resolution        =���     t  ]x   TEMP                	   
_FillValue        G�O�   	long_name         $Sea temperature in-situ ITS-90 scale   standard_name         sea_water_temperature      units         degree_Celsius     	valid_min         �      	valid_max         B      C_format      %9.3f      FORTRAN_format        F9.3   
resolution        :�o     t  k�   TEMP_QC                    
_FillValue               	long_name         quality flag   conventions       Argo reference table 2       �  z`   TEMP_ADJUSTED                   	   
_FillValue        G�O�   	long_name         $Sea temperature in-situ ITS-90 scale   standard_name         sea_water_temperature      units         degree_Celsius     	valid_min         �      	valid_max         B      C_format      %9.3f      FORTRAN_format        F9.3   
resolution        :�o     t  ~    TEMP_ADJUSTED_QC                   
_FillValue               	long_name         quality flag   conventions       Argo reference table 2       �  �t   TEMP_ADJUSTED_ERROR                    
_FillValue        G�O�   	long_name         VContains the error on the adjusted values as determined by the delayed mode QC process     units         degree_Celsius     C_format      %9.3f      FORTRAN_format        F9.3   
resolution        :�o     t  �   PSAL                	   
_FillValue        G�O�   	long_name         Practical salinity     standard_name         sea_water_salinity     units         psu    	valid_min         @      	valid_max         B$     C_format      %9.3f      FORTRAN_format        F9.3   
resolution        :�o     t  ��   PSAL_QC                    
_FillValue               	long_name         quality flag   conventions       Argo reference table 2       �  ��   PSAL_ADJUSTED                   	   
_FillValue        G�O�   	long_name         Practical salinity     standard_name         sea_water_salinity     units         psu    	valid_min         @      	valid_max         B$     C_format      %9.3f      FORTRAN_format        F9.3   
resolution        :�o     t  ��   PSAL_ADJUSTED_QC                   
_FillValue               	long_name         quality flag   conventions       Argo reference table 2       �  �   PSAL_ADJUSTED_ERROR                    
_FillValue        G�O�   	long_name         VContains the error on the adjusted values as determined by the delayed mode QC process     units         psu    C_format      %9.3f      FORTRAN_format        F9.3   
resolution        :�o     t  °   	PARAMETER                            
_FillValue               	long_name         /List of parameters with calibration information    conventions       Argo reference table 3        0  �$   SCIENTIFIC_CALIB_EQUATION                   	         
_FillValue               	long_name         'Calibration equation for this parameter         �T   SCIENTIFIC_CALIB_COEFFICIENT                	         
_FillValue               	long_name         *Calibration coefficients for this equation          �T   SCIENTIFIC_CALIB_COMMENT                	         
_FillValue               	long_name         .Comment applying to this parameter calibration          �T   SCIENTIFIC_CALIB_DATE                   
         
_FillValue               	long_name         Date of calibration    conventions       YYYYMMDDHHMISS        ,  �T   HISTORY_INSTITUTION                       
_FillValue               	long_name         "Institution which performed action     conventions       Argo reference table 4          ڀ   HISTORY_STEP                      
_FillValue               	long_name         Step in data processing    conventions       Argo reference table 12         ڄ   HISTORY_SOFTWARE                      
_FillValue               	long_name         'Name of software which performed action    conventions       Institution dependent           ڈ   HISTORY_SOFTWARE_RELEASE                      
_FillValue               	long_name         2Version/release of software which performed action     conventions       Institution dependent           ڌ   HISTORY_REFERENCE                         
_FillValue               	long_name         Reference of database      conventions       Institution dependent         @  ڐ   HISTORY_DATE             
         
_FillValue               	long_name         #Date the history record was created    conventions       YYYYMMDDHHMISS          ��   HISTORY_ACTION                        
_FillValue               	long_name         Action performed on data   conventions       Argo reference table 7          ��   HISTORY_PARAMETER                         
_FillValue               	long_name         (Station parameter action is performed on   conventions       Argo reference table 3          ��   HISTORY_START_PRES                     
_FillValue        G�O�   	long_name          Start pressure action applied on   units         decibar         ��   HISTORY_STOP_PRES                      
_FillValue        G�O�   	long_name         Stop pressure action applied on    units         decibar         ��   HISTORY_PREVIOUS_VALUE                     
_FillValue        G�O�   	long_name         +Parameter/Flag previous value before action         ��   HISTORY_QCTEST                        
_FillValue               	long_name         <Documentation of tests performed, tests failed (in hex form)   conventions       EWrite tests performed when ACTION=QCP$; tests failed when ACTION=QCF$           � Argo profile    3.1 1.2 19500101000000  20220324003141  20220324003141  1901765 Argo Australia                                                  Peter Oke                                                       PRES            TEMP            PSAL               A   CS  1901765/1                       2B  A   NAVIS_EBR                       1323                            ARGO 170425                     869 @��A�fO1   @��B�d���8\(��@SW{J#9�1   GPS     A   A   A   Primary sampling: averaged []                                                                                                                                                                                                                                      @�  @���@���A   A@  A`  A�  A�  A�  A�  A�  A�  A�  A�  A�33B  B  B  B   B(  B0  B8  B@  BH  BP  BX  B`  BhffBpffBw��B�  B�  B�33B�  B���B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  C   C  C  C  C  C
  C  C  C  C  C  C  C  C  C  C  C   C"  C$  C&  C(  C*  C,  C.  C0  C2  C4  C6  C8  C:  C<  C>  C@  CB  CD  CF  CH  CJ  CL  CN  CP  CR  CT  CV  CX  CZ  C\  C^  C`  Cb  Cd  Cf  Ch  Cj  Cl  Cn  Cp  Cr  Ct  Cv  Cx  Cz�C|  C~  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C��3C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  D   D � D  D� D  D� D  D� D  D� D  D� D  D� D  D� D  D� D	  D	� D
  D
� D  D� D  D� D  D� D  D� D  D� D  D� D  D� D  D� D  D� D  D� D  D� D  D� D  D� D  D� D  D� D  D� D  D� D  D� D  D� D  D� D  D� D   D � D!  D!� D"  D"� D#  D#� D$  D$� D%  D%� D&  D&� D'  D'� D(  D(� D)  D)� D*  D*� D+  D+� D,  D,� D-  D-� D.  D.� D/  D/� D0  D0� D1  D1� D2  D2� D3  D3� D4  D4� D5  D5� D6  D6� D7  D7� D8  D8� D9  D9� D:  D:� D;  D;� D<  D<� D=  D=� D>  D>� D?  D?� D@  D@� DA  DA� DB  DB� DC  DC� DD  DD� DE  DE� DF  DF� DG  DG� DH  DH� DI  DI� DJ  DJ� DK  DK� DL  DL� DM  DM� DN  DN� DO  DO� DP  DP� DQ  DQ� DR  DR� DS  DS� DT  DT� DU  DU� DV  DV� DW  DW� DX  DX� DY  DY� DZ  DZ� D[  D[� D\  D\� D]  D]� D^  D^� D_  D_� D`  D`� Da  Day�Db  Db� Dc  Dc� Dd  Dd� De  De� Df  Df� Dg  Dg� Dh  Dh� Di  Di� Dj  Dj� Dk  Dk� Dl  Dl� Dm  Dm� Dn  Dn� Do  Do� Dp  Dp� Dq  Dq� Dr  Dr� Ds  Ds� Dt  Dt� Du  Du� Dv  Dv� Dw  Dw� Dx  Dx� Dy  Dy� Dz  Dz� D{  D{� D|  D|� D}  D}� D~  D~� D  D� D�  D�@ D�� D�� D�  D�@ D�� D�� D�  D�@ D�� D�� D�  D�@ D�� D�� D�  D�@ D�� D�� D�  D�@ D�� D�� D�  D�@ D�� D�� D�  D�@ D�� D�� D�  D�@ D�� D�� D�  D�@ D�� D�� D�  D�@ D�� D�� D�  D�@ D�� D�� D�  D�@ D�� D�� D�  D�@ D�� D�� D�  D�@ D�� D�� D�  D�@ D�� D�� D�  D�@ D�� D�� D�  D�@ D�� D�� D�  D�@ D�� D�� D�  D�@ D�� D�� D�  D�<�D�� D�� D�  D�@ D�� D�� D�  D�@ D�� D�� D�  D�@ D�� D�� D�  D�@ D�� D�� D�  D�@ D�� D�� D�  D�@ D�� D�� D�  D�C3D�� D�� D�  D�@ D�� D�� D�  D�@ D�� D�� D�  D�@ D�� D�� D�  D�@ D�� D�� D�  D�@ D�� D�� D�  D�@ D�� D�� D�  D�@ D�� D�� D�  D�@ D�� D�� D�  D�@ D�� D�� D�  D�@ D�� D�� D�  D�@ D�� D�� D�  D�@ D�� D�� D�  D�@ D�� D�� D�  D�@ D�� D�� D�  D�@ D�� D�� D�  D�<�D�� D�� D�  D�@ D�� D�� D�  D�@ D�� D�� D�  D�@ D�� D�� D�  D�@ D�� D�� D�  D�@ D�� D�� D�  D�@ D�� D�� D�  D�@ D�� D�� D�  D�@ D�� D�� D�  D�@ D�� D�� D�  D�@ D�� D�� D�  D�@ D�� D�� D�  D�@ D�� D�� D�  D�@ D�� D�� D�  D�@ D�� D�� D�  D�@ D�� D�� D�  D�@ D�� D�� D�  D�@ D�� D�� D�  D�@ D�� D�� D�  D�@ D�� D�� D�  D�@ D�� D�� D�  D�@ D�� D�� D�  D�@ D�� D�� D�  D�@ D D�� D�  D�@ DÀ D�� D�  D�@ DĀ D�� D�  D�@ Dŀ D�� D�  D�@ Dƀ D�� D�  D�@ Dǀ D�� D�  D�@ DȀ D�� D�  D�@ Dɀ D�� D�  D�@ Dʀ D�� D�3D�@ Dˀ D�� D�  D�@ D̀ D�� D�  D�@ D̀ D�� D�  D�@ D΀ D�� D�  D�@ Dπ D�� D�  D�@ DЀ D�� D�  D�@ Dр D�� D�  D�@ DҀ D�� D�  D�@ DӀ D�� D�  D�@ DԀ D�� D�  D�@ DՀ Dռ�D�  D�@ Dր D�� D�  D�@ D׀ D�� D�  D�@ D؀ D�� D�  D�@ Dـ D�� D�  D�@ Dڀ D�� D�  D�@ Dۀ D�� D�  D�C3D܀ D�� D�  D�@ D݀ D�� D�  D�@ Dހ D�� D�  D�@ D߀ D�� D�  D�@ D�� D�� D�  D�@ D� D�� D�  D�@ D� D�� D�  D�@ D� D�� D�  D�@ D� D�� D�  D�@ D� D�� D�  D�@ D� D�� D�  D�@ D� 1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111   @�Q�@��A ��A$(�AD(�Ad(�A�{A�{A�{A�{A�{A�{A�{A�{B ��B	
=B
=B
=B!
=B)
=B1
=B9
=BA
=BI
=BQ
=BY
=Ba
=Bip�Bqp�Bx��B��B��B��RB��B�Q�B��B��B��B��B��B��B��B��B��B��B��B��BąBȅB̅BЅBԅB؅B܅B��B�B�B�B��B�B��B��C B�CB�CB�CB�CB�C
B�CB�CB�CB�CB�CB�CB�CB�CB�CB�CB�C B�C"B�C$B�C&B�C(B�C*B�C,B�C.B�C0B�C2B�C4B�C6B�C8B�C:B�C<B�C>B�C@B�CBB�CDB�CFB�CHB�CJB�CLB�CNB�CPB�CRB�CTB�CVB�CXB�CZB�C\B�C^B�C`B�CbB�CdB�CfB�ChB�CjB�ClB�CnB�CpB�CrB�CtB�CvB�CxB�Cz\)C|B�C~B�C�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�{C�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HC�!HD �D ��D�D��D�D��D�D��D�D��D�D��D�D��D�D��D�D��D	�D	��D
�D
��D�D��D�D��D�D��D�D��D�D��D�D��D�D��D�D��D�D��D�D��D�D��D�D��D�D��D�D��D�D��D�D��D�D��D�D��D�D��D�D��D�D��D �D ��D!�D!��D"�D"��D#�D#��D$�D$��D%�D%��D&�D&��D'�D'��D(�D(��D)�D)��D*�D*��D+�D+��D,�D,��D-�D-��D.�D.��D/�D/��D0�D0��D1�D1��D2�D2��D3�D3��D4�D4��D5�D5��D6�D6��D7�D7��D8�D8��D9�D9��D:�D:��D;�D;��D<�D<��D=�D=��D>�D>��D?�D?��D@�D@��DA�DA��DB�DB��DC�DC��DD�DD��DE�DE��DF�DF��DG�DG��DH�DH��DI�DI��DJ�DJ��DK�DK��DL�DL��DM�DM��DN�DN��DO�DO��DP�DP��DQ�DQ��DR�DR��DS�DS��DT�DT��DU�DU��DV�DV��DW�DW��DX�DX��DY�DY��DZ�DZ��D[�D[��D\�D\��D]�D]��D^�D^��D_�D_��D`�D`��Da�Da�>Db�Db��Dc�Dc��Dd�Dd��De�De��Df�Df��Dg�Dg��Dh�Dh��Di�Di��Dj�Dj��Dk�Dk��Dl�Dl��Dm�Dm��Dn�Dn��Do�Do��Dp�Dp��Dq�Dq��Dr�Dr��Ds�Ds��Dt�Dt��Du�Du��Dv�Dv��Dw�Dw��Dx�Dx��Dy�Dy��Dz�Dz��D{�D{��D|�D|��D}�D}��D~�D~��D�D��D�RD�HRD��RD��RD�RD�HRD��RD��RD�RD�HRD��RD��RD�RD�HRD��RD��RD�RD�HRD��RD��RD�RD�HRD��RD��RD�RD�HRD��RD��RD�RD�HRD��RD��RD�RD�HRD��RD��RD�RD�HRD��RD��RD�RD�HRD��RD��RD�RD�HRD��RD��RD�RD�HRD��RD��RD�RD�HRD��RD��RD�RD�HRD��RD��RD�RD�HRD��RD��RD�RD�HRD��RD��RD�RD�HRD��RD��RD�RD�HRD��RD��RD�RD�HRD��RD��RD�RD�ED��RD��RD�RD�HRD��RD��RD�RD�HRD��RD��RD�RD�HRD��RD��RD�RD�HRD��RD��RD�RD�HRD��RD��RD�RD�HRD��RD��RD�RD�K�D��RD��RD�RD�HRD��RD��RD�RD�HRD��RD��RD�RD�HRD��RD��RD�RD�HRD��RD��RD�RD�HRD��RD��RD�RD�HRD��RD��RD�RD�HRD��RD��RD�RD�HRD��RD��RD�RD�HRD��RD��RD�RD�HRD��RD��RD�RD�HRD��RD��RD�RD�HRD��RD��RD�RD�HRD��RD��RD�RD�HRD��RD��RD�RD�HRD��RD��RD�RD�ED��RD��RD�RD�HRD��RD��RD�RD�HRD��RD��RD�RD�HRD��RD��RD�RD�HRD��RD��RD�RD�HRD��RD��RD�RD�HRD��RD��RD�RD�HRD��RD��RD�RD�HRD��RD��RD�RD�HRD��RD��RD�RD�HRD��RD��RD�RD�HRD��RD��RD�RD�HRD��RD��RD�RD�HRD��RD��RD�RD�HRD��RD��RD�RD�HRD��RD��RD�RD�HRD��RD��RD�RD�HRD��RD��RD�RD�HRD��RD��RD�RD�HRD��RD��RD�RD�HRD��RD��RD�RD�HRD��RD��RD�RD�HRD��RD��RD�RD�HRDRD��RD�RD�HRDÈRD��RD�RD�HRDĈRD��RD�RD�HRDňRD��RD�RD�HRDƈRD��RD�RD�HRDǈRD��RD�RD�HRDȈRD��RD�RD�HRDɈRD��RD�RD�HRDʈRD��RD��D�HRDˈRD��RD�RD�HRD̈RD��RD�RD�HRD͈RD��RD�RD�HRDΈRD��RD�RD�HRDψRD��RD�RD�HRDЈRD��RD�RD�HRDшRD��RD�RD�HRD҈RD��RD�RD�HRDӈRD��RD�RD�HRDԈRD��RD�RD�HRDՈRD��D�RD�HRDֈRD��RD�RD�HRD׈RD��RD�RD�HRD؈RD��RD�RD�HRDوRD��RD�RD�HRDڈRD��RD�RD�HRDۈRD��RD�RD�K�D܈RD��RD�RD�HRD݈RD��RD�RD�HRDވRD��RD�RD�HRD߈RD��RD�RD�HRD��RD��RD�RD�HRD�RD��RD�RD�HRD�RD��RD�RD�HRD�RD��RD�RD�HRD�RD��RD�RD�HRD�RD��RD�RD�HRD�RD��RD�RD�HRD�R1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111   G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�A�Q�A�XA�ZA�`BA�`BA�dZA�bNA�`BA�dZA�dZA�Q�A�?}A�/A�"�A�$�A�XA��#AǁA�XA��A�ZA�-A��A���A��#AŲ-A�r�A�
=A�O�A�%A�K�A�ZA�`BA���A���A��jA�A���A�E�A�ffA��
A�{A�ĜA���A���A�`BA�p�A��A���A�&�A�x�A���A���A���A�bA�Q�A���A��+A��A�`BA�A��DA��A�Q�A��`A��^A�ZA��A�t�A�XA�ffA�-A�  A�&�A���A��A�S�A�S�A���A��A��!A�O�A���A��PA��A���A��A�/A�ƨA�C�A�z�A�VA��TA�|�A�A��9A��+A�n�A���A��PA�5?A��`A��PAA~(�A|��A{�TAz{Ay�Ax��Ax�`AxȴAx��Aw�TAv1Atz�Asx�ArA�Ap�yAo��Am�TAmdZAl~�Ak��Ak&�Aj��Aj{Ah��Ag�hAf=qAd-Ac�PAb��AbA�Aa&�A`1A^�yA^bA]ƨA]`BA\�!A[�7AZZAY�hAY+AX��AW��AW��AWp�AW?}AV�jAV$�AU�AUO�AT��AS��ASK�AR��AR�DAR  AQ��AQC�AP��APn�APE�AO��AO?}AO�AN�AN(�AM�;AM"�ALM�AK�^AKt�AKhsAKO�AK
=AJ�yAJȴAJE�AI�#AIhsAIG�AIVAH�jAHv�AHbAG�hAG;dAGVAF�/AF~�AE�mAEC�AEAD��ADr�AC�AC��ACS�AB�HABQ�AB�AA�
AA��AA33A@�9A@�uA@I�A?ƨA?��A?dZA?VA>��A>�\A>ZA>$�A>  A=��A=�^A=�PA=K�A=&�A<��A<E�A;ƨA;`BA;?}A;VA:�RA:n�A:-A9�A9�A9+A8�9A8M�A7��A7�hA77LA6�A6��A6�\A6jA6  A5�hA5"�A4��A4ZA4 �A3�TA3��A3S�A2r�A1�A1x�A17LA0�A0��A0v�A/�
A/��A/hsA.�yA.ZA.JA-�PA-%A,$�A+|�A+p�A+S�A+;dA*�jA*9XA)ƨA)7LA(��A(��A(bA'�A'�wA'��A'O�A&�/A&Q�A%�mA%�^A%p�A$�A$VA#�#A#�PA#K�A"�yA"��A"A�A!��A!l�A �A ��A -AAp�A
=A�A��A�\A�+A{A�AhsA�A�+A1'A��A��A?}A�A�jAffAA�A�A~�A�A�#A��A�PA|�AdZA�`Az�AA�A��A�A�PAl�A�AJAp�AG�A��A5?A��AO�A��A�!AjA�TA��A+A�`Ar�AA�A�
Ap�A33AVA��A-A��A
ȴA
jA
$�A
  A	hsA	33A�HA�Ax�A�Az�A{A�-AG�AZA�wAx�AC�A
=A��AE�A�mA�AA r�A 1'@��@��+@�/@�ƨ@��@��P@�@�&�@�I�@�;d@�$�@�@�j@�+@�&�@�b@�!@���@�j@��@��@�dZ@柾@��@�t�@�+@��D@�l�@�@��@�bN@�|�@�$�@�`B@�9X@��@�ff@�/@ԛ�@��m@ѩ�@�(�@�"�@�M�@̣�@�o@�%@�S�@�`B@Ĭ@�(�@�C�@��7@�ƨ@���@�5?@��h@��@��w@�ȴ@��@�V@�bN@�  @��P@�\)@�~�@��@��;@�ȴ@�J@��h@�r�@�(�@�1@�t�@���@���@���@��@�V@�%@�r�@���@��@���@�=q@��#@��@��;@�@���@�ff@���@�Z@���@���@�J@�Ĝ@�j@���@�t�@�;d@�$�@��@�?}@��D@�r�@�1'@���@���@��@��T@�X@���@��@�b@��w@��@�l�@�;d@���@�{@�x�@���@�  @�dZ@�ȴ@��@���@���@��@��@���@��m@�l�@�+@��y@��!@�n�@�=q@��@���@�@�/@��9@��@�9X@��@���@�t�@���@�ff@���@��7@�7L@�V@��/@��@�bN@�A�@�1'@�b@��
@��@�|�@�o@�M�@�$�@�@��@��-@�x�@�`B@�G�@�%@��9@��@� �@l�@
=@~��@~��@~�@~��@}�@}?}@|��@|j@|1@{�
@{��@{S�@z�H@y��@yhs@y%@xĜ@x��@xr�@x1'@w�@w�@v�+@u�@uO�@tI�@t�@s��@r��@rn�@rJ@q��@qhs@q7L@q7L@q&�@q%@pr�@pbN@o�;@o�P@o;d@o+@o�@n��@nȴ@nv�@nff@n5?@m�@m�@m/@m�@l��@l��@lZ@k�F@k�@kt�@kdZ@k33@j�@j�!@j^5@j=q@jJ@iG�@hA�@g�@gl�@g;d@g
=@fȴ@f��@f��@e�T@e�@e?}@d�@d�j@d�@d�D@dj@dI�@c��@c��@cdZ@c33@b�H@b�\@b�@a�@a��@a7L@`��@`��@`��@`A�@_��@_�P@_\)@^�@^�R@^��@^$�@]�@]��@]��@]�h@]�@]`B@^{@]�T@]@]�h@]`B@\�@\Z@\�@[ƨ@[C�@[o@Z^5@Y��@Y�^@Y&�@YG�@Y�7@YX@YX@Y�7@Yhs@Y�@X1'@W�w@Wl�@W|�@W\)@W+@W�@V��@V�@V{@U��@U�-@U�@UV@T�@T�@T(�@T9X@SC�@S"�@So@R��@R��@R�@Rn�@R~�@R��@R��@Q�^@Q��@R-@R^5@R-@Q��@QX@QX@Q��@R�@RJ@Q�^@P�u@P�u@O�w@O+@OK�@N��@N��@Nff@M�-@L�/@L�@K��@KC�@K"�@J��@J�\@JM�@J-@JJ@I��@I��@I��@Ix�@Ihs@Ihs@I�#@I��@H��@H  @G�@G|�@G\)@G
=@G
=@F��@G
=@G
=@F�R@FE�@F{@E��@E�h@E`B@E?}@D��@D�@D�j@D�D@Dz�@Dj@D9X@D1@D1@C��@C�m@C�
@Cƨ@C�F@C�@C"�@C@B�@B��@Bn�@Bn�@Bn�@Bn�@B^5@B=q@A��@A�#@A�#@A��@A�7@AX@AG�@A7L@A�@@��@@Ĝ@@��@@�@@b@?�@?�@?|�@?\)@?\)@?K�@>�y@>�R@>V@>E�@>5?@>$�@>$�@>{@>{@>{@>ff@>�y@?;d@?�@?
=@>�+@=p�@=O�@=`B@=`B@=@=�@=�@=�@=�-@=O�@<��@<�/@<��@<��@<��@<��@<�@<z�@<1@;�m@;t�@;"�@;o@;@:��@:�\@:M�@:�@9�@9�@9��@9x�@9X@9&�@8��@8 �@7�@7�P@7�@6�@6��@6V@6{@5�h@5O�@4��@4�D@4Z@4�@3�F@3o@2M�@2�@1��@1�#@1��@1�7@0Ĝ@0  @/\)@.��@.�+@.v�@.5?@-�-@,��@,j@+ƨ@+�@+dZ@+33@+@+@*�@*�H@*�!@)x�1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111   A�Q�A�XA�ZA�`BA�`BA�dZA�bNA�`BA�dZA�dZA�Q�A�?}A�/A�"�A�$�A�XA��#AǁA�XA��A�ZA�-A��A���A��#AŲ-A�r�A�
=A�O�A�%A�K�A�ZA�`BA���A���A��jA�A���A�E�A�ffA��
A�{A�ĜA���A���A�`BA�p�A��A���A�&�A�x�A���A���A���A�bA�Q�A���A��+A��A�`BA�A��DA��A�Q�A��`A��^A�ZA��A�t�A�XA�ffA�-A�  A�&�A���A��A�S�A�S�A���A��A��!A�O�A���A��PA��A���A��A�/A�ƨA�C�A�z�A�VA��TA�|�A�A��9A��+A�n�A���A��PA�5?A��`A��PAA~(�A|��A{�TAz{Ay�Ax��Ax�`AxȴAx��Aw�TAv1Atz�Asx�ArA�Ap�yAo��Am�TAmdZAl~�Ak��Ak&�Aj��Aj{Ah��Ag�hAf=qAd-Ac�PAb��AbA�Aa&�A`1A^�yA^bA]ƨA]`BA\�!A[�7AZZAY�hAY+AX��AW��AW��AWp�AW?}AV�jAV$�AU�AUO�AT��AS��ASK�AR��AR�DAR  AQ��AQC�AP��APn�APE�AO��AO?}AO�AN�AN(�AM�;AM"�ALM�AK�^AKt�AKhsAKO�AK
=AJ�yAJȴAJE�AI�#AIhsAIG�AIVAH�jAHv�AHbAG�hAG;dAGVAF�/AF~�AE�mAEC�AEAD��ADr�AC�AC��ACS�AB�HABQ�AB�AA�
AA��AA33A@�9A@�uA@I�A?ƨA?��A?dZA?VA>��A>�\A>ZA>$�A>  A=��A=�^A=�PA=K�A=&�A<��A<E�A;ƨA;`BA;?}A;VA:�RA:n�A:-A9�A9�A9+A8�9A8M�A7��A7�hA77LA6�A6��A6�\A6jA6  A5�hA5"�A4��A4ZA4 �A3�TA3��A3S�A2r�A1�A1x�A17LA0�A0��A0v�A/�
A/��A/hsA.�yA.ZA.JA-�PA-%A,$�A+|�A+p�A+S�A+;dA*�jA*9XA)ƨA)7LA(��A(��A(bA'�A'�wA'��A'O�A&�/A&Q�A%�mA%�^A%p�A$�A$VA#�#A#�PA#K�A"�yA"��A"A�A!��A!l�A �A ��A -AAp�A
=A�A��A�\A�+A{A�AhsA�A�+A1'A��A��A?}A�A�jAffAA�A�A~�A�A�#A��A�PA|�AdZA�`Az�AA�A��A�A�PAl�A�AJAp�AG�A��A5?A��AO�A��A�!AjA�TA��A+A�`Ar�AA�A�
Ap�A33AVA��A-A��A
ȴA
jA
$�A
  A	hsA	33A�HA�Ax�A�Az�A{A�-AG�AZA�wAx�AC�A
=A��AE�A�mA�AA r�A 1'@��@��+@�/@�ƨ@��@��P@�@�&�@�I�@�;d@�$�@�@�j@�+@�&�@�b@�!@���@�j@��@��@�dZ@柾@��@�t�@�+@��D@�l�@�@��@�bN@�|�@�$�@�`B@�9X@��@�ff@�/@ԛ�@��m@ѩ�@�(�@�"�@�M�@̣�@�o@�%@�S�@�`B@Ĭ@�(�@�C�@��7@�ƨ@���@�5?@��h@��@��w@�ȴ@��@�V@�bN@�  @��P@�\)@�~�@��@��;@�ȴ@�J@��h@�r�@�(�@�1@�t�@���@���@���@��@�V@�%@�r�@���@��@���@�=q@��#@��@��;@�@���@�ff@���@�Z@���@���@�J@�Ĝ@�j@���@�t�@�;d@�$�@��@�?}@��D@�r�@�1'@���@���@��@��T@�X@���@��@�b@��w@��@�l�@�;d@���@�{@�x�@���@�  @�dZ@�ȴ@��@���@���@��@��@���@��m@�l�@�+@��y@��!@�n�@�=q@��@���@�@�/@��9@��@�9X@��@���@�t�@���@�ff@���@��7@�7L@�V@��/@��@�bN@�A�@�1'@�b@��
@��@�|�@�o@�M�@�$�@�@��@��-@�x�@�`B@�G�@�%@��9@��@� �@l�@
=@~��@~��@~�@~��@}�@}?}@|��@|j@|1@{�
@{��@{S�@z�H@y��@yhs@y%@xĜ@x��@xr�@x1'@w�@w�@v�+@u�@uO�@tI�@t�@s��@r��@rn�@rJ@q��@qhs@q7L@q7L@q&�@q%@pr�@pbN@o�;@o�P@o;d@o+@o�@n��@nȴ@nv�@nff@n5?@m�@m�@m/@m�@l��@l��@lZ@k�F@k�@kt�@kdZ@k33@j�@j�!@j^5@j=q@jJ@iG�@hA�@g�@gl�@g;d@g
=@fȴ@f��@f��@e�T@e�@e?}@d�@d�j@d�@d�D@dj@dI�@c��@c��@cdZ@c33@b�H@b�\@b�@a�@a��@a7L@`��@`��@`��@`A�@_��@_�P@_\)@^�@^�R@^��@^$�@]�@]��@]��@]�h@]�@]`B@^{@]�T@]@]�h@]`B@\�@\Z@\�@[ƨ@[C�@[o@Z^5@Y��@Y�^@Y&�@YG�@Y�7@YX@YX@Y�7@Yhs@Y�@X1'@W�w@Wl�@W|�@W\)@W+@W�@V��@V�@V{@U��@U�-@U�@UV@T�@T�@T(�@T9X@SC�@S"�@So@R��@R��@R�@Rn�@R~�@R��@R��@Q�^@Q��@R-@R^5@R-@Q��@QX@QX@Q��@R�@RJ@Q�^@P�u@P�u@O�w@O+@OK�@N��@N��@Nff@M�-@L�/@L�@K��@KC�@K"�@J��@J�\@JM�@J-@JJ@I��@I��@I��@Ix�@Ihs@Ihs@I�#@I��@H��@H  @G�@G|�@G\)@G
=@G
=@F��@G
=@G
=@F�R@FE�@F{@E��@E�h@E`B@E?}@D��@D�@D�j@D�D@Dz�@Dj@D9X@D1@D1@C��@C�m@C�
@Cƨ@C�F@C�@C"�@C@B�@B��@Bn�@Bn�@Bn�@Bn�@B^5@B=q@A��@A�#@A�#@A��@A�7@AX@AG�@A7L@A�@@��@@Ĝ@@��@@�@@b@?�@?�@?|�@?\)@?\)@?K�@>�y@>�R@>V@>E�@>5?@>$�@>$�@>{@>{@>{@>ff@>�y@?;d@?�@?
=@>�+@=p�@=O�@=`B@=`B@=@=�@=�@=�@=�-@=O�@<��@<�/@<��@<��@<��@<��@<�@<z�@<1@;�m@;t�@;"�@;o@;@:��@:�\@:M�@:�@9�@9�@9��@9x�@9X@9&�@8��@8 �@7�@7�P@7�@6�@6��@6V@6{@5�h@5O�@4��@4�D@4Z@4�@3�F@3o@2M�@2�@1��@1�#@1��@1�7@0Ĝ@0  @/\)@.��@.�+@.v�@.5?@-�-@,��@,j@+ƨ@+�@+dZ@+33@+@+@*�@*�H@*�!@)x�1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111   G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�B��B��B��B��B��B��B��B��B��B��B��B�B�B�B�BI�B^5BjBq�B�%B�\B�{B��B��B��B�9B�jB��B��B��B��B��B��B�B�B�B�mB�B��B��B�B�B�B��B+B\BoBuB{BoB\BoB�B�B�B�B�B�B,B2-B49B33B33B49B49B49B33B2-B0!B33B/B.B-B(�B#�B!�B �B�BoB
=B%BB��B��B��B��B�B�;B�B��B�FB��B��B��B�uB�\B�PB�DB�Br�Bm�BhsBcTB[#BO�BF�B?}B2-B(�B'�B&�B&�B$�B�B{BDBB��B�B�B�;B�#B��B��BɺBŢB��B�FB�B��B�PB�%B~�By�Bo�Be`B[#BR�BN�BK�BD�B:^B1'B(�B$�B �B�B�B�B{BbB
=B	7BB  B��B�B�B�B�fB�NB�;B�B�
B�B��B��B��BȴBÖB��B�XB�3B�B�B��B��B��B��B��B��B��B��B��B�{B�oB�VB�JB�+B�B�B� B}�Bx�Br�Bo�Bm�Bk�BffBdZB`BB]/BXBT�BR�BP�BL�BG�BF�BC�B?}B=qB<jB9XB5?B49B1'B0!B.B-B,B)�B'�B&�B#�B�B�B�BoBoB\BJB
=B+BBB��B��B�B�B�B�B�yB�sB�fB�TB�5B�#B�B��B��B��B��BǮB��B�dB�LB�?B�-B�'B�B��B��B��B��B��B��B�hB�JB�%B~�B}�B|�B{�Bw�Br�Bn�BiyBffBdZB_;B]/B[#BZBXBS�BN�BJ�BH�BF�BB�B=qB8RB5?B33B0!B-B)�B%�B"�B�B�B�BuBbBVBJB
=B	7B	7BBB
��B
��B
��B
��B
�B
�B
�B
�B
�sB
�`B
�NB
�5B
�B
�B
��B
��B
��B
��B
��B
��B
ǮB
ÖB
B
�}B
�qB
�jB
�dB
�LB
�-B
�B
��B
��B
��B
��B
��B
��B
��B
�uB
�bB
�VB
�DB
�7B
�+B
�B
�B
~�B
|�B
z�B
x�B
v�B
q�B
m�B
jB
hsB
gmB
e`B
cTB
aHB
]/B
YB
VB
S�B
P�B
N�B
K�B
H�B
D�B
B�B
A�B
@�B
>wB
<jB
:^B
8RB
5?B
2-B
1'B
/B
,B
)�B
%�B
$�B
#�B
"�B
�B
�B
�B
�B
�B
uB
\B
DB
1B
%B
B
B	��B	��B	��B	�B	�B	�B	�B	�fB	�TB	�BB	�5B	�/B	�#B	�B	�B	�B	��B	��B	��B	��B	��B	ȴB	ŢB	ÖB	��B	�}B	�jB	�XB	�LB	�3B	�-B	�'B	�!B	�B	�B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	�B	�B	�B	�'B	�'B	�-B	�3B	�?B	�FB	�RB	�^B	�dB	�dB	�dB	�jB	�jB	�qB	�wB	�}B	��B	��B	B	B	ĜB	ŢB	ƨB	ǮB	ȴB	ȴB	ɺB	��B	��B	��B	��B	��B	��B	��B	�B	�
B	�
B	�B	�B	�B	�B	�#B	�/B	�BB	�HB	�NB	�NB	�TB	�ZB	�`B	�`B	�mB	�sB	�yB	�B	�B	�B	�B	�B	�B	�B	�B	��B	��B	��B	��B	��B	��B	��B	��B	��B
  B
  B
B
B
B
B
B
+B
	7B
DB
PB
bB
bB
hB
oB
oB
uB
�B
�B
�B
�B
�B
�B
�B
�B
�B
�B
�B
�B
�B
�B
�B
�B
�B
�B
 �B
!�B
!�B
#�B
#�B
$�B
%�B
'�B
(�B
(�B
(�B
)�B
)�B
,B
-B
-B
.B
.B
.B
0!B
0!B
0!B
1'B
2-B
2-B
2-B
49B
5?B
5?B
7LB
7LB
7LB
8RB
8RB
8RB
:^B
;dB
;dB
<jB
<jB
=qB
=qB
>wB
>wB
?}B
@�B
@�B
A�B
B�B
C�B
D�B
D�B
F�B
F�B
F�B
H�B
H�B
I�B
L�B
N�B
O�B
P�B
T�B
XB
ZB
[#B
]/B
]/B
^5B
^5B
_;B
`BB
aHB
`BB
aHB
cTB
dZB
ffB
iyB
hsB
hsB
jB
k�B
k�B
l�B
l�B
l�B
m�B
p�B
p�B
r�B
s�B
s�B
t�B
t�B
u�B
v�B
v�B
v�B
u�B
v�B
w�B
v�B
w�B
x�B
x�B
y�B
z�B
{�B
{�B
|�B
}�B
}�B
}�B
� B
�B
�B
�B
�B
�B
�+B
�1B
�7B
�1B
�+B
�1B
�+B
�+B
�7B
�7B
�7B
�7B
�7B
�=B
�=B
�=B
�=B
�DB
�JB
�JB
�JB
�JB
�JB
�PB
�VB
�VB
�VB
�\B
�\B
�oB
�{B
�uB
�uB
�uB
�{B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
�B
�B
�B
�B
�B
�B
�B
�!B
�'B
�'B
�-B
�-B
�3B
�9B
�9B
�9B
�?B
�?B
�?B
�?B
�FB
�FB
�LB
�LB
�LB
�LB
�LB
�RB
�XB
�^B
�^B
�^B
�dB
�dB
�dB
�jB
�jB
�qB
�}B
��B
��B
��B
��B
��B
��B
��B
B
ĜB
ǮB
ǮB
ǮB
ǮB
ȴB
ȴB
ȴB
ȴB
ȴB
ȴB
ȴB
ȴB
ȴB
ȴB
ȴB
ȴB
ȴB
ȴB
ȴB
ȴB
ɺB
ɺB
ɺB
ɺB
ɺB
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111   B��B��B��B��B��B��B��B��B��B��B��B�B�B�B�BI�B^5BjBq�B�%B�\B�{B��B��B��B�9B�jB��B��B��B��B��B��B�B�B�B�mB�B��B��B�B�B�B��B+B\BoBuB{BoB\BoB�B�B�B�B�B�B,B2-B49B33B33B49B49B49B33B2-B0!B33B/B.B-B(�B#�B!�B �B�BoB
=B%BB��B��B��B��B�B�;B�B��B�FB��B��B��B�uB�\B�PB�DB�Br�Bm�BhsBcTB[#BO�BF�B?}B2-B(�B'�B&�B&�B$�B�B{BDBB��B�B�B�;B�#B��B��BɺBŢB��B�FB�B��B�PB�%B~�By�Bo�Be`B[#BR�BN�BK�BD�B:^B1'B(�B$�B �B�B�B�B{BbB
=B	7BB  B��B�B�B�B�fB�NB�;B�B�
B�B��B��B��BȴBÖB��B�XB�3B�B�B��B��B��B��B��B��B��B��B��B�{B�oB�VB�JB�+B�B�B� B}�Bx�Br�Bo�Bm�Bk�BffBdZB`BB]/BXBT�BR�BP�BL�BG�BF�BC�B?}B=qB<jB9XB5?B49B1'B0!B.B-B,B)�B'�B&�B#�B�B�B�BoBoB\BJB
=B+BBB��B��B�B�B�B�B�yB�sB�fB�TB�5B�#B�B��B��B��B��BǮB��B�dB�LB�?B�-B�'B�B��B��B��B��B��B��B�hB�JB�%B~�B}�B|�B{�Bw�Br�Bn�BiyBffBdZB_;B]/B[#BZBXBS�BN�BJ�BH�BF�BB�B=qB8RB5?B33B0!B-B)�B%�B"�B�B�B�BuBbBVBJB
=B	7B	7BBB
��B
��B
��B
��B
�B
�B
�B
�B
�sB
�`B
�NB
�5B
�B
�B
��B
��B
��B
��B
��B
��B
ǮB
ÖB
B
�}B
�qB
�jB
�dB
�LB
�-B
�B
��B
��B
��B
��B
��B
��B
��B
�uB
�bB
�VB
�DB
�7B
�+B
�B
�B
~�B
|�B
z�B
x�B
v�B
q�B
m�B
jB
hsB
gmB
e`B
cTB
aHB
]/B
YB
VB
S�B
P�B
N�B
K�B
H�B
D�B
B�B
A�B
@�B
>wB
<jB
:^B
8RB
5?B
2-B
1'B
/B
,B
)�B
%�B
$�B
#�B
"�B
�B
�B
�B
�B
�B
uB
\B
DB
1B
%B
B
B	��B	��B	��B	�B	�B	�B	�B	�fB	�TB	�BB	�5B	�/B	�#B	�B	�B	�B	��B	��B	��B	��B	��B	ȴB	ŢB	ÖB	��B	�}B	�jB	�XB	�LB	�3B	�-B	�'B	�!B	�B	�B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	��B	�B	�B	�B	�'B	�'B	�-B	�3B	�?B	�FB	�RB	�^B	�dB	�dB	�dB	�jB	�jB	�qB	�wB	�}B	��B	��B	B	B	ĜB	ŢB	ƨB	ǮB	ȴB	ȴB	ɺB	��B	��B	��B	��B	��B	��B	��B	�B	�
B	�
B	�B	�B	�B	�B	�#B	�/B	�BB	�HB	�NB	�NB	�TB	�ZB	�`B	�`B	�mB	�sB	�yB	�B	�B	�B	�B	�B	�B	�B	�B	��B	��B	��B	��B	��B	��B	��B	��B	��B
  B
  B
B
B
B
B
B
+B
	7B
DB
PB
bB
bB
hB
oB
oB
uB
�B
�B
�B
�B
�B
�B
�B
�B
�B
�B
�B
�B
�B
�B
�B
�B
�B
�B
 �B
!�B
!�B
#�B
#�B
$�B
%�B
'�B
(�B
(�B
(�B
)�B
)�B
,B
-B
-B
.B
.B
.B
0!B
0!B
0!B
1'B
2-B
2-B
2-B
49B
5?B
5?B
7LB
7LB
7LB
8RB
8RB
8RB
:^B
;dB
;dB
<jB
<jB
=qB
=qB
>wB
>wB
?}B
@�B
@�B
A�B
B�B
C�B
D�B
D�B
F�B
F�B
F�B
H�B
H�B
I�B
L�B
N�B
O�B
P�B
T�B
XB
ZB
[#B
]/B
]/B
^5B
^5B
_;B
`BB
aHB
`BB
aHB
cTB
dZB
ffB
iyB
hsB
hsB
jB
k�B
k�B
l�B
l�B
l�B
m�B
p�B
p�B
r�B
s�B
s�B
t�B
t�B
u�B
v�B
v�B
v�B
u�B
v�B
w�B
v�B
w�B
x�B
x�B
y�B
z�B
{�B
{�B
|�B
}�B
}�B
}�B
� B
�B
�B
�B
�B
�B
�+B
�1B
�7B
�1B
�+B
�1B
�+B
�+B
�7B
�7B
�7B
�7B
�7B
�=B
�=B
�=B
�=B
�DB
�JB
�JB
�JB
�JB
�JB
�PB
�VB
�VB
�VB
�\B
�\B
�oB
�{B
�uB
�uB
�uB
�{B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
�B
�B
�B
�B
�B
�B
�B
�!B
�'B
�'B
�-B
�-B
�3B
�9B
�9B
�9B
�?B
�?B
�?B
�?B
�FB
�FB
�LB
�LB
�LB
�LB
�LB
�RB
�XB
�^B
�^B
�^B
�dB
�dB
�dB
�jB
�jB
�qB
�}B
��B
��B
��B
��B
��B
��B
��B
B
ĜB
ǮB
ǮB
ǮB
ǮB
ȴB
ȴB
ȴB
ȴB
ȴB
ȴB
ȴB
ȴB
ȴB
ȴB
ȴB
ȴB
ȴB
ȴB
ȴB
ȴB
ɺB
ɺB
ɺB
ɺB
ɺB
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��B
��1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111   G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�G�O�PRES            TEMP            PSAL            Pcorrected = Praw - surface offset                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              This sensor is subject to hysteresis                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            2022032400312620220324003130                CS  ARFMPYRTv0.1                                                                20220324003015  IP                  G�O�G�O�G�O�                CS  ARGQPYRTv0.1                                                                20220324003015  QCP$                G�O�G�O�G�O�208FB7E         CS  ARGQPYRTv0.1                                                                20220324003015  QCF$                G�O�G�O�G�O�0               