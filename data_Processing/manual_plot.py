import matplotlib.pyplot as plt

sgd_train = [0.3946494, 0.43655357, 0.45026305, 0.47289363, 0.49744567, 0.5199085, 0.5403889, 0.5636599,
             0.59060615, 0.625467, 0.66441476, 0.71112466, 0.7570263, 0.80381244, 0.83762103, 0.86879146,
             0.8980557, 0.91719407, 0.93212354, 0.94372857, 0.9533664, 0.9626687, 0.9672741, 0.9724285,
             0.97744566, 0.9798246, 0.9814106, 0.9814411, 0.98357606, 0.9851315, 0.987907, 0.98722076,
             0.9899504, 0.9891422, 0.99327487, 0.9927259, 0.99324435, 0.9940374, 0.9907434, 0.99522686,
             0.9953946, 0.99402213, 0.99306136, 0.99350363, 0.99550134, 0.9985665, 0.9980785, 0.99458635,
             0.99540985, 0.9965078]

sgd_val = [0.41399336, 0.42146018, 0.45768806, 0.48230088, 0.5011062, 0.52571905, 0.5409292, 0.5528208,
           0.5608407, 0.5575221, 0.57992256, 0.59457964, 0.5829646, 0.5862832, 0.585177, 0.60896015, 0.59457964,
           0.6020465, 0.6125553, 0.60868365, 0.6285951, 0.6225111, 0.6214049, 0.63384956, 0.6498894, 0.6482301,
           0.6465708, 0.6308075, 0.6490597, 0.647677, 0.6368916, 0.64961284, 0.642146, 0.6670354, 0.65486723,
           0.6568031, 0.6706305, 0.6670354, 0.66288716, 0.6653761, 0.6642699, 0.6609513, 0.6598451, 0.64629424,
           0.6664823, 0.67809737, 0.6816925, 0.65099555, 0.6601217, 0.66150445]

adam_train = [0.42019817, 0.47725505, 0.5370797, 0.58542126, 0.63052994, 0.6718109, 0.7128936, 0.746382, 0.77903163,
              0.8077468, 0.826809, 0.8472894, 0.86845595, 0.8783988, 0.8912238, 0.9042928, 0.91525733, 0.9220282,
              0.9274114, 0.933679, 0.9421883, 0.94691575, 0.94982845, 0.9544796, 0.9562486, 0.95849025, 0.96440715,
              0.96544415, 0.96669465, 0.96666414, 0.97130007, 0.9721998, 0.9728555, 0.97332823, 0.97645444, 0.97662216,
              0.9773847, 0.9802974, 0.9808006, 0.9801144, 0.9824171, 0.9801601, 0.9821273, 0.985589, 0.9837743,
              0.9834388, 0.98443, 0.98627526, 0.9861228, 0.98400307]

adam_val = [0.44469026, 0.49751106, 0.54065263, 0.56913716, 0.5901549, 0.6073009, 0.616427, 0.60259956, 0.60536504,
            0.6095133, 0.61117256, 0.6147677, 0.6120022, 0.60978985, 0.6167035, 0.59070796, 0.5887721, 0.5976217,
            0.59292036, 0.61615044, 0.6081305, 0.60481197, 0.6178097, 0.6070243, 0.61725664, 0.62057525, 0.6175332,
            0.61974555, 0.6056416, 0.5926438, 0.5915376, 0.60868365, 0.6114491, 0.6106195, 0.6056416, 0.59568584,
            0.6186394, 0.6175332, 0.6186394, 0.60038716, 0.6136615, 0.60536504, 0.59928095, 0.5998341, 0.61283183,
            0.5998341, 0.6045354, 0.60757744, 0.5948562, 0.6067478]

rmsprop_train = [0.42082316, 0.5053145, 0.5457415, 0.57502097, 0.5968433, 0.6132367, 0.6281662, 0.63967973, 0.6522455,
                 0.66179186, 0.6721769, 0.6785208, 0.6904308, 0.68820435, 0.6926725, 0.69778115, 0.71734655, 0.7174838,
                 0.7171483, 0.72930235, 0.73704916, 0.7465345, 0.7514297, 0.75452536, 0.7626839, 0.7710408, 0.76889056,
                 0.7677926, 0.77467024, 0.77650017, 0.78836447, 0.7896149, 0.801281, 0.8030042, 0.8028822, 0.8091956,
                 0.8112238, 0.8126268, 0.82525355, 0.827724, 0.826138, 0.8328479, 0.8357606, 0.83762103, 0.83983225,
                 0.8445902, 0.84469694, 0.84295845, 0.8397865, 0.8471521]

rmsprop_val = [0.45298672, 0.54728985, 0.5658186, 0.57190263, 0.5807522, 0.5931969, 0.5931969, 0.576604, 0.5757743,
               0.5948562, 0.57411504, 0.59292036, 0.5813053, 0.5577987, 0.58102876, 0.5909845, 0.5835177, 0.5824115,
               0.59292036, 0.6020465, 0.60536504, 0.58600664, 0.57356197, 0.55724555, 0.5445243, 0.60591817, 0.5948562,
               0.58711284, 0.60425884, 0.5995575, 0.5790929, 0.6020465, 0.5782633, 0.5887721, 0.59928095, 0.5707965,
               0.55033183, 0.54397124, 0.6017699, 0.59679204, 0.5876659, 0.5738385, 0.57466817, 0.5807522, 0.6009403,
               0.5954093, 0.5890487, 0.58268803, 0.58821905, 0.46321902]

plt.plot(sgd_train, label='SGD_train')
plt.plot(sgd_val, label='SGD_val')
plt.plot(rmsprop_train, label='RMSprop_train', marker='x')
plt.plot(rmsprop_val, label='RMSprop_val', marker='x')
plt.plot(adam_train, label='Adam_train', marker='.')
plt.plot(adam_val, label='Adam_val', marker='.')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.4, 1])
plt.legend(loc='upper left')

plt.grid(True)
plt.show()

sgd_train = [0.66243064, 0.70840824, 0.7165483, 0.7211232, 0.7239312, 0.72787505, 0.7317558, 0.7354157, 0.73913866,
             0.7425146, 0.7480675, 0.751191, 0.7545985, 0.75888944, 0.76305413, 0.7674712, 0.77412844, 0.7801546,
             0.7863701, 0.7928695, 0.8016091, 0.8079192, 0.81706893, 0.82653415, 0.8397539, 0.8504496, 0.86297524,
             0.8737656, 0.89023507, 0.8956933, 0.90487456, 0.91304624, 0.92153335, 0.9285376, 0.93377507, 0.9403691,
             0.9481622, 0.9499606, 0.95437765, 0.95942575, 0.96179205, 0.9648525, 0.97087866, 0.9702792, 0.97229844,
             0.97397065, 0.9758952, 0.97889256, 0.97778827, 0.9785455]

sgd_val = [0.6897727, 0.6670455, 0.65170455, 0.6715909, 0.7130682, 0.6931818, 0.7096591, 0.71079546, 0.72386366,
           0.71875, 0.7335227, 0.7278409, 0.7318182, 0.72329545, 0.73295456, 0.72897726, 0.7443182, 0.7443182,
           0.7039773, 0.74715906, 0.74488634, 0.7426136, 0.72670454, 0.73977274, 0.7335227, 0.7596591, 0.7465909,
           0.75284094, 0.75, 0.7227273, 0.74375, 0.7125, 0.71363634, 0.7403409, 0.72897726, 0.76193184, 0.7568182,
           0.7647727, 0.725, 0.76931816, 0.7534091, 0.76193184, 0.7596591, 0.77102274, 0.77670455, 0.7568182, 0.7704545,
           0.74147725, 0.7721591, 0.7477273]

adam_train = [0.6757379, 0.7149708, 0.7236157, 0.7392018, 0.7513803, 0.7604039, 0.7694589, 0.7798706, 0.7930273,
              0.7996214, 0.81179994, 0.8225272, 0.83754534, 0.850071, 0.8592838, 0.87480676, 0.8834201, 0.89402115,
              0.89976335, 0.9038965, 0.9131409, 0.9200189, 0.9285692, 0.9263922, 0.93320715, 0.9370563, 0.93664616,
              0.9422622, 0.945228, 0.94806755, 0.9513804, 0.9535574, 0.9565231, 0.9546301, 0.95854235, 0.9599621,
              0.9589841, 0.9632434, 0.9685755, 0.96633536, 0.96598834, 0.96655625, 0.9700899, 0.97289795, 0.97359204,
              0.9740022, 0.9750118, 0.9757375, 0.9783878, 0.9778829]

adam_val = [0.64602274, 0.66988635, 0.6943182, 0.7403409, 0.7477273, 0.7375, 0.74886364, 0.7607955, 0.76193184,
            0.7653409, 0.7653409, 0.75795454, 0.75454545, 0.7403409, 0.7318182, 0.7443182, 0.76363635, 0.7556818,
            0.7477273, 0.74545455, 0.7318182, 0.74375, 0.73238635, 0.75454545, 0.75852275, 0.7517046, 0.75795454,
            0.74147725, 0.7409091, 0.74715906, 0.7431818, 0.74147725, 0.7465909, 0.73636365, 0.76022726, 0.73238635,
            0.75454545, 0.7607955, 0.74488634, 0.7613636, 0.7465909, 0.72329545, 0.74375, 0.74545455, 0.74147725,
            0.7443182, 0.74886364, 0.75454545, 0.74147725, 0.7261364]

rmsprop_train = [0.66952574, 0.71411896, 0.72711784, 0.7379397, 0.74421835, 0.75292635, 0.7569333, 0.7622653, 0.766714,
                 0.7669664, 0.764253, 0.7657359, 0.76393753, 0.7660514, 0.7698375, 0.7663985, 0.7676921, 0.76917493,
                 0.76835465, 0.7737498, 0.76974285, 0.7708156, 0.7685755, 0.7705632, 0.769522, 0.7733081, 0.7702792,
                 0.77466476, 0.7753273, 0.7780092, 0.78192145, 0.77627385, 0.7718568, 0.7722354, 0.77393913, 0.77229846,
                 0.7749487, 0.77141505, 0.77340275, 0.760183, 0.77214074, 0.7635274, 0.7578798, 0.75431454, 0.76141346,
                 0.7615081, 0.7613819, 0.75797445, 0.7663985, 0.7423253]

rmsprop_val = [0.6988636, 0.6943182, 0.69261366, 0.71079546, 0.7477273, 0.7125, 0.7255682, 0.7255682, 0.62954545,
               0.7125, 0.7255682, 0.7346591, 0.73977274, 0.73125, 0.71761364, 0.7153409, 0.73977274, 0.72897726,
               0.6659091, 0.7539773, 0.70454544, 0.74147725, 0.7056818, 0.68863636, 0.7096591, 0.74488634, 0.71988636,
               0.7261364, 0.7164773, 0.7130682, 0.7386364, 0.69943184, 0.67897725, 0.7318182, 0.71875, 0.7426136,
               0.7255682, 0.6, 0.58181816, 0.6653409, 0.74375, 0.5647727, 0.7443182, 0.75113636, 0.71420455, 0.6693182,
               0.66136366, 0.71079546, 0.66136366, 0.6982955]

plt.plot(sgd_train, label='SGD_train')
plt.plot(sgd_val, label='SGD_val')
plt.plot(rmsprop_train, label='RMSprop_train', marker='x')
plt.plot(rmsprop_val, label='RMSprop_val', marker='x')
plt.plot(adam_train, label='Adam_train', marker='.')
plt.plot(adam_val, label='Adam_val', marker='.')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower center')

plt.grid(True)
plt.show()

sgd_train = [0.795274, 0.8241012, 0.8420456, 0.8538353, 0.8626099, 0.8703034, 0.87708193, 0.8838812, 0.8905142,
             0.89825964, 0.9034267, 0.9148317, 0.9241574, 0.9360198, 0.9384526, 0.95457757, 0.9573327, 0.9677396,
             0.9708586, 0.9775539, 0.98514336, 0.9868172, 0.98819995, 0.9885534, 0.99288875, 0.9931487, 0.99137086,
             0.9948849, 0.9964756, 0.9973073, 0.9973073, 0.9965795, 0.9967875, 0.99795187, 0.9987628, 0.9976504,
             0.9980974, 0.99797267, 0.99779594, 0.9972345, 0.9963924, 0.99772316, 0.9987108, 0.9982118, 0.9994802,
             0.9998856, 0.9999376, 0.9995945, 0.9999896, 1.0]

sgd_val = [0.81006736, 0.7995883, 0.8504865, 0.8604042, 0.8654566, 0.86938626, 0.8727545, 0.8800524, 0.8836078,
           0.8875374, 0.8819237, 0.8733159, 0.8729416, 0.8851048, 0.8699476, 0.8789296, 0.8727545, 0.8701347, 0.8785554,
           0.86470807, 0.869012, 0.87462574, 0.87743264, 0.8849177, 0.87649703, 0.88042665, 0.8819237, 0.8895958,
           0.88922155, 0.8845434, 0.8867889, 0.88529193, 0.8867889, 0.8866018, 0.88697606, 0.8877246, 0.8894087,
           0.8845434, 0.8907186, 0.8817365, 0.8880988, 0.8938997, 0.8817365, 0.8877246, 0.89296407, 0.89483535,
           0.89502245, 0.8953967, 0.8952096, 0.89502245]

adam_train = [0.8220746, 0.8621109, 0.88106376, 0.89548373, 0.907315, 0.9201235, 0.9284927, 0.938411, 0.94740397,
              0.9551078, 0.9602749, 0.96575385, 0.96971494, 0.9736344, 0.97794896, 0.9785, 0.98136944, 0.9828977,
              0.98419726, 0.9864221, 0.98789847, 0.98883414, 0.9894787, 0.99035203, 0.99000895, 0.99116296, 0.9916932,
              0.99216104, 0.9927848, 0.9938245, 0.99360615, 0.99414676, 0.9937621, 0.99446905, 0.9949369, 0.9945626,
              0.9939388, 0.9960597, 0.9951552, 0.99506164, 0.9960909, 0.9964444, 0.9962053, 0.99587256, 0.9963716,
              0.9962261, 0.99627805, 0.996434, 0.9966523, 0.99664193]

adam_val = [0.8536677, 0.8787425, 0.89127994, 0.8916542, 0.88267213, 0.8856662, 0.88398206, 0.8895958, 0.89801645,
            0.8847305, 0.8982036, 0.89202845, 0.8895958, 0.88342065, 0.8935254, 0.8903443, 0.8905314, 0.89502245,
            0.8909057, 0.8997006, 0.89408684, 0.8944611, 0.8918413, 0.8955838, 0.8924027, 0.89277697, 0.8903443,
            0.88342065, 0.8895958, 0.88641465, 0.89202845, 0.88622755, 0.8897829, 0.88416916, 0.8856662, 0.8879117,
            0.8894087, 0.8905314, 0.8905314, 0.89876497, 0.88529193, 0.89408684, 0.88622755, 0.88922155, 0.8982036,
            0.8888473, 0.8938997, 0.8895958, 0.8894087, 0.8909057]

rmsprop_train = [0.8153173, 0.8568919, 0.8658225, 0.86807853, 0.8690766, 0.8738798, 0.87205, 0.8705737, 0.8723099,
                 0.86855674, 0.86563534, 0.85732853, 0.85921025, 0.8579419, 0.85299313, 0.85303473, 0.854064,
                 0.83840686, 0.8290084, 0.81674045, 0.7963633, 0.7868609, 0.8274281, 0.8310669, 0.76810557, 0.73640656,
                 0.81834155, 0.75967395, 0.7639989, 0.7442455, 0.5938494, 0.5879546, 0.5962094, 0.6103799, 0.6127711,
                 0.6019587, 0.6136132, 0.60724014, 0.6077911, 0.616098, 0.6130726, 0.60362214, 0.6077496, 0.60379887,
                 0.61351967, 0.60810304, 0.60418355, 0.6131973, 0.60658514, 0.60515046]

rmsprop_val = [0.83177394, 0.8358907, 0.88117516, 0.87238026, 0.86583084, 0.877994, 0.88622755, 0.8633982, 0.8721931,
               0.8701347, 0.86938626, 0.8488024, 0.8409431, 0.88323355, 0.88061374, 0.8748129, 0.8205464, 0.7297904,
               0.619012, 0.8059506, 0.85797155, 0.8787425, 0.85872006, 0.87256736, 0.5097305, 0.61676645, 0.82466316,
               0.67234284, 0.59711826, 0.5301272, 0.502994, 0.5116018, 0.54247755, 0.5770958, 0.6085329, 0.5479042,
               0.528256, 0.54752994, 0.5778443, 0.559506, 0.55707335, 0.5692365, 0.53574103, 0.5662425, 0.5604416,
               0.5668039, 0.5623129, 0.5660554, 0.56212574, 0.5589446]

plt.plot(sgd_train, label='SGD_train')
plt.plot(sgd_val, label='SGD_val')
plt.plot(rmsprop_train, label='RMSprop_train', marker='x')
plt.plot(rmsprop_val, label='RMSprop_val', marker='x')
plt.plot(adam_train, label='Adam_train', marker='.')
plt.plot(adam_val, label='Adam_val', marker='.')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='right')

plt.grid(True)
plt.show()