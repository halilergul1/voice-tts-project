# LLM rater: gemini-3.5-flash-lite

299 rated clips heard without their transcript; labelled 292 (the rest: other / no answer); accuracy on the labelled 0.4966. Tokens used: 43126.

## gemini-3.5-flash-lite

| class | predicted | correct | precision | rated | recall |
|---|---|---|---|---|---|
| angry | 97 | 43 | 0.4433 | 64 | 0.6719 |
| happy | 31 | 11 | 0.3548 | 30 | 0.3667 |
| sad | 49 | 15 | 0.3061 | 45 | 0.3333 |
| neutral | 115 | 76 | 0.6609 | 151 | 0.5033 |

Precision when only answers at or above a self-reported confidence are kept (correct / kept):

| class | >= 0.6 | >= 0.7 | >= 0.8 | >= 0.85 | >= 0.9 | >= 0.95 | >= 0.97 | >= 0.99 |
|---|---|---|---|---|---|---|---|---|
| angry | 0.44 (43/97) | 0.44 (43/97) | 0.46 (43/94) | 0.45 (35/77) | 0.46 (21/46) | 0.50 (3/6) | - | - |
| sad | 0.31 (15/49) | 0.31 (15/49) | 0.34 (14/41) | 0.34 (10/29) | 0.44 (8/18) | 0.67 (2/3) | - | - |
| happy | 0.35 (11/31) | 0.35 (11/31) | 0.40 (10/25) | 0.39 (7/18) | 0.38 (5/13) | 1.00 (1/1) | - | - |

Confusion (predicted -> rated): angry->angry 43, angry->happy 7, angry->neutral 35, angry->other 4, angry->sad 8, happy->angry 1, happy->happy 11, happy->neutral 17, happy->sad 2, neutral->angry 11, neutral->happy 7, neutral->neutral 76, neutral->other 1, neutral->other/sad 1, neutral->sad 19, sad->angry 9, sad->happy 2, sad->neutral 21, sad->other 2, sad->sad 15

## emotion2vec+ on the same clips (default cut)

| class | predicted | correct | precision | rated | recall |
|---|---|---|---|---|---|
| angry | 60 | 51 | 0.85 | 64 | 0.7969 |
| happy | 59 | 27 | 0.4576 | 30 | 0.9 |
| sad | 60 | 37 | 0.6167 | 45 | 0.8222 |
| neutral | 60 | 56 | 0.9333 | 151 | 0.3709 |

## Agreement: the two labelled the same clip 233 times and agreed on 90 (39%)

## Delivery words from the LLM (first 40)

- 8bORUKi26Cc_00126: calm conversation (neutral, rated sad)
- L0Txo5QR7Rw_02031_x2: loud, forceful, exasperated delivery (angry, rated other)
- 5Q4eUJ7d5RM_00955: shouting aggressively (angry, rated neutral)
- ndssE4SMOaA_01392: casual speaking (neutral, rated neutral)
- L0Txo5QR7Rw_02468: complaining and agitated voice (angry, rated neutral)
- L0Txo5QR7Rw_00869_x3: concerned and thoughtful speech (sad, rated neutral)
- zSLNUwcbqqo_01879: calm delivery (neutral, rated neutral)
- GENdU9GxZ9g_00534: calm speech (neutral, rated neutral)
- tLq-42uUldg_00220: neutral speech (neutral, rated sad)
- L0Txo5QR7Rw_02225: firm command (angry, rated other)
- GENdU9GxZ9g_00283: neutral statement (neutral, rated angry)
- 5Q4eUJ7d5RM_00495: matter of fact (neutral, rated neutral)
- xrcd4wca6aU_00016: neutral delivery (neutral, rated neutral)
- ntgfc01JNJY_00485: calm explanation (neutral, rated neutral)
- ndssE4SMOaA_00952: lively speech (happy, rated neutral)
- tLq-42uUldg_00031: pleading tone (sad, rated neutral)
- L0Txo5QR7Rw_02428: serious question (neutral, rated neutral)
- ntgfc01JNJY_00745: firm statement (neutral, rated angry)
- tLq-42uUldg_00452: matter of fact (neutral, rated neutral)
- tLq-42uUldg_00212: casual remark (neutral, rated sad)
- ntgfc01JNJY_01766: concerned speech (sad, rated sad)
- L0Txo5QR7Rw_01873: harsh accusation (angry, rated neutral)
- xrcd4wca6aU_00547: shouting speech (angry, rated angry)
- GENdU9GxZ9g_01050: firm demand (angry, rated neutral)
- ndssE4SMOaA_01427: casual narration (neutral, rated neutral)
- zSLNUwcbqqo_00509: warning threat (angry, rated angry)
- esrefruya_ep007_00410: calm statement (neutral, rated neutral)
- 5Q4eUJ7d5RM_00452: explaining plan (neutral, rated neutral)
- xrcd4wca6aU_00570_x2: serious plea (angry, rated angry)
- tLq-42uUldg_00337: narrating a story (neutral, rated happy)
- ntgfc01JNJY_00342: complaining tone (angry, rated neutral)
- L0Txo5QR7Rw_00850_x2: calm narration (neutral, rated neutral)
- xrcd4wca6aU_01130: scolding angrily (angry, rated angry)
- ndssE4SMOaA_01987: serious delivery (sad, rated angry)
- GENdU9GxZ9g_00987: serious speech (neutral, rated neutral)
- esrefruya_ep010_00990: casual speaking (neutral, rated neutral)
- zSLNUwcbqqo_01077: dramatic expression (other, rated other)
- zSLNUwcbqqo_01050: gloomy and soft (sad, rated sad)
- 8bORUKi26Cc_00288: grateful tone (happy, rated neutral)
- GENdU9GxZ9g_00052: dramatic speech (neutral, rated neutral)
