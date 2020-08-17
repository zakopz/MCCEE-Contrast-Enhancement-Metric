
%%% Read original and enhanced images
orig_image = imread('original_im.bmp');
if(size(orig_image,3)==3)
    original =  rgb2gray(orig_image);
else
    original =  orig_image;
end

enhanced_image = imread('enhanced_im.bmp'); 
if(size(orig_image,3)==3)
    enhanced = rgb2gray(enhanced_image);
else
    enhanced =  enhanced_image);
end   

%%% Evaluate Steerable Pyramid Wavelet transforms for original and enhanced images
cd toolbox_wavelets
options.nb_orientations = 2;
spyr = perform_steerable_transform(double(original),log2(size(original,1))-2,options);
A11 = spyr{1,6};
O11 = spyr{1,2};
O21 = spyr{1,3};
O12 = spyr{1,4};
O22 = spyr{1,5};

spyr_enh = perform_steerable_transform(double(enhanced),log2(size(enhanced,1))-2,options);
A11e = spyr_enh{1,6};
O11e = spyr_enh{1,2};
O21e = spyr_enh{1,3};
O12e = spyr_enh{1,4};
O22e = spyr_enh{1,5};

%%% Evaluate four metrics for the four criteria
smo_o2_lev2 = SMO(uint8(O22),uint8(O22e)); % SMO for O22 subband
lom_approx = LOM(uint8(A11),uint8(A11e));  % LOM for approx subband
ambe = AMBE(original, enhanced);  % AMBE
emee = emee(double(enhanced),8,1); % EMEE

feature_vector = [lom_approx ambe smo_o2_lev2 emee];

load('model_q_ceed')
%%% Use the trained model with LibSVM library to predict the MCCEE metric for the enhanced image
[mccee_val,acc_q,prob_q] = svmpredict(0.5, feature_vector, model_q, '-q');