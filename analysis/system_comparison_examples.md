# MedJarGone: Manual Analysis — System Comparison Examples

Three systems compared on clinical case reports from MultiClinSum test set (N=3,396).

| System | Description |
|--------|-------------|
| **D2** | BioBART-v2-large LoRA fine-tuned (`max_new_tokens=512`) |
| **Qwen** | Qwen2.5-3B-Instruct zero-shot, no fine-tuning |
| **D3** | D2 output → MLM jargon detection → Qwen rewrite |

**Full test-set metrics (N=3,396):**

| System | ROUGE-1 | ROUGE-2 | BLEU | BERTScore F1 | FK Grade ↓ | SARI ↑ |
|--------|---------|---------|------|:------------:|:----------:|:------:|
| D2     | 36.68   | 16.54   | 12.25 | 0.8547      | 13.05      | 36.53  |
| D3     | 33.92   | 12.86   | 8.98  | 0.8508      | 11.92      | 30.63  |
| Qwen   | 33.45   | 9.22    | 5.19  | 0.8536      | 11.61      | 40.53  |

---

## Example groups

| Group | Theme | Examples |
|-------|-------|----------|
| **A** | High jargon + D2 FK >> Qwen FK | 1–6 |
| **B** | D3 achieves lowest FK — rewrite working | 7–10 |
| **C** | Qwen most readable; compare completeness vs D2 | 11–12 |
| **D** | D3 degraded vs D2 — pipeline failure mode | 13 |
| **E** | Long source (800+ words) — all systems under pressure | 14–16 |

---

## Example 1 &nbsp;·&nbsp; ID 2794 &nbsp;·&nbsp; Group A

> High jargon load, large FK gap — D2 much harder to read than Qwen

| Metric | D2 | Qwen | D3 | Gold |
|--------|:--:|:----:|:--:|:----:|
| FK Grade | 26.3 | 10.8 | 21.7 | 9.1 |
| Word count | 159 | 113 | 171 | 34 |
| Source words | 323 | | | |

**MLM jargon spans detected (16 total, top 12 shown):**
`partum`, `oral antidiabetic medication`, `left flank pain`, `hematuria`, `one week’s duration`, `palpation`, `lumbar contact`, `an ectatic left renal vein`, `seat`, `a large hypodense thrombus`, `the inferior vena cava`, `a large heterogeneous and hypodense region`

<details>
<summary><strong>Source (clinical case report)</strong></summary>

A 43-year-old Caucasian woman, a mother of three sons, at 30 days post-partum after a full-term pregnancy without any complications and concluded by a vaginal delivery, suffering from diabetes on oral antidiabetic medication, presented to the emergency department with left flank pain with hematuria of one week’s duration. A clinical examination revealed an alert patient, febrile at 38°C, with blood pressure of 158 over 67mmHg, heart rate of 90 beats per minute, respiratory rate of 20 breaths per minute, suffering from pain in the left flank (visual analog score (VAS) = 5), this pain was exacerbated by palpation, which found lumbar contact. Diagnostic studies in the emergency room revealed a urine analysis with 104 white blood cells (WBCs), gross hematuria, and a C-reactive protein level of 88. Her blood cell count revealed anemia at 9g of hemoglobin; blood electrolytes and renal function with serum creatinine levels were normal. An abdominal ultrasound scan showed discrete pyelocalyceal cavities dilation with a slightly enlarged left kidney and without visible urinary obstruction. A CT urography and angiography scan was performed and showed an ectatic left renal vein, seat of a large hypodense thrombus extending from the segmental renal veins to the inferior vena cava, with a large heterogeneous and hypodense region located in the medium and lower poles of the left kidney without urinary tract dilatation or tumoral lesions; this aspect concluded in a perfusion abnormality (renal venous infarction) (Figures and ). Anticoagulation heparin therapy was started and our patient was admitted to a

*[source truncated at 250 words — full length: 323 words]*

</details>

<details>
<summary><strong>Gold reference summary</strong></summary>

We report the case of a 43-year-old Caucasian woman, a mother of three sons who presented with left flank pain and hematuria. The clinical investigations did not find any other cause for her thrombophilia.

</details>

**D2 — BioBART LoRA** *(FK 26.3)*

A 43-year-old Caucasian woman, a mother of three sons, at 30 days post-partum after a full-term pregnancy without any complications and concluded by a vaginal delivery, suffering from diabetes on oral antidiabetic medication, presented to the emergency department with left flank pain with hematuria of one week’s duration. A clinical examination revealed an alert patient, febrile at 38°C, with blood pressure of 158 over 67mmHg, heart rate of 90 beats per minute, respiratory rate of 20 breaths per minute; this pain was exacerbated by palpation, which found lumbar contact. A CT urography and angiography scan was performed and showed an ectatic left renal vein, seat of a large hypodense thrombus extending from the segmental renal veins through the inferior vena cava, with a large heterogeneous and hypodense region located in the medium and lower poles of the left kidney without urinary tract dilatation or tumoral lesions; that aspect concluded in a perfusion abnormality (renal venous infarction) (Figures and fixme).

**Qwen-only** *(FK 10.8)*

A 43-year-old woman with diabetes presented to the emergency room complaining of left flank pain and red-colored urine. Her examination showed fever, tenderness in the left lower abdomen area. Tests revealed high counts of white blood cells, abnormal blood pressure, and anemia. An abdominal ultrasound and CT urography showed issues with her kidney veins and kidneys. She was diagnosed with renal vein thrombosis causing a blocked path for blood flow in her left kidney, which had become inflamed (renal venous infarction). Treatment included anticoagulant therapy with heparin first, then oral acenocoumarol once she improved. The pain went away after five days, and she recovered fully and was sent home from the medical center.

**D3 — BioBART→MedJEx→Qwen rewrite** *(FK 21.7)*

A 43-year-old Caucasian woman, a mother of three sons, at 30 days post-partum after a full-term pregnancy without any complications and concluded by a vaginal delivery, suffering from diabetes on oral antidiabetic medication, presented to the emergency department with left flank pain and hematuria for one week. A clinical examination revealed an alert patient, febrile at 38°C, with blood pressure of 158 over 67mmHg, heart rate of 90 beats per minute, respiratory rate of 20 breaths per minute; this pain was exacerbated by palpation, which found discomfort around the lower back area. Imaging studies, including a CT urography and angiography, showed a widened left renal vein as the location of a large hypodense thrombus extending from the segmental veins into the inferior vena cava. There was also a large region with abnormal density located in the mid-to-lower regions of the left kidney, which did not show signs of urinary tract dilation or tumour lesions; these findings suggested a lack of adequate blood flow to part of the kidney (renal venous infarction).

### Analysis

| | D2 | Qwen | D3 |
|---|---|---|---|
| **Strengths** | | | |
| **Weaknesses** | | | |
| **Factual accuracy** | | | |
| **Readability** | | | |
| **Completeness** | | | |

**What a better system would do here:**

<!-- fill in -->

---

## Example 2 &nbsp;·&nbsp; ID 2793 &nbsp;·&nbsp; Group A

> High jargon load, large FK gap — D2 much harder to read than Qwen

| Metric | D2 | Qwen | D3 | Gold |
|--------|:--:|:----:|:--:|:----:|
| FK Grade | 22.4 | 8.2 | 25.0 | 17.5 |
| Word count | 150 | 152 | 130 | 68 |
| Source words | 186 | | | |

**MLM jargon spans detected (26 total, top 12 shown):**
`nonconsaguine young and healthy parents`, `a hypotrophic infant`, `intrauterine growth retardation`, `asphyxia`, `Complex phenotypic dysmorphic features`, `The phenotype`, `our patient`, `midface retrusion`, `frontal bossing`, `downslanted palpebral fissures`, `large filtrum`, `micrognathia`

<details>
<summary><strong>Source (clinical case report)</strong></summary>

A 15-year-old boy suffering from SWS was born from the third pregnancy of nonconsaguine young and healthy parents as a hypotrophic infant with a perinatal risk factor which included intrauterine growth retardation and asphyxia. Complex phenotypic dysmorphic features were observed at birth. The phenotype of our patient includes multiple facial and skeletal disorders: midface retrusion, prominent forehead, frontal bossing, shallow orbits, downslanted palpebral fissures, narrow nasal bridge, large filtrum, micrognathia, small hands with short metacarpal bones and short distal phalanges of fingers with consequent brachydactyly, knee flexion contracture, pseudoarthrosis, calcaneovalgus deformity, thoracic scoliosis, lumbar hyperlordosis, coxa valga, skeletal dysplasia, pectus carinatum, generalized hypotonia, global developmental delay, cognitive impairment, complete lack of adipose tissue with consequent pseudohypertrophy of muscles, bilateral cataracts and progeroid facial appearance ( and ). As a part of the syndrome, an extremely short stature, body weight of 15 kg and body height of 100 cm were observed.
Computed tomography (CT) scan of cranio-cervical junction revealed bifid arch of the atlas as a part of skeletal disorders. Additional preoperative magnetic resonance imaging (MRI) revealed a critical stenosis of cranio-cervical junction with concomitant myelopathy .

</details>

<details>
<summary><strong>Gold reference summary</strong></summary>

We present a case of a 15-years-old boy with clinical and radiological characteristics of SWS. Genetic examination identified a pathogenic heterozygous variant in the COG4 gene. Magnetic resonance imaging revealed a critical stenosis of the cranio-cervical junction (CCJ) which required surgical treatment to attempt sufficient neurological decompression. The patient underwent decompression of CCJ under general anesthesia. There was no significant radiological and clinical improvement during the postoperative period.

</details>

**D2 — BioBART LoRA** *(FK 22.4)*

A 15-year-old boy suffering from SWS was born from the third pregnancy of nonconsaguine young and healthy parents as a hypotrophic infant with a perinatal risk factor which included intrauterine growth retardation and asphyxia. Complex phenotypic dysmorphic features were observed at birth. The phenotype of our patient includes multiple facial and skeletal disorders: midface retrusion, prominent forehead, frontal bossing, shallow orbits, downslanted palpebral fissures, narrow nasal bridge, large filtrum, micrognathia, small hands with short metacarpal bones and short distal phalanges of fingers with consequent brachydactyly, knee flexion contracture, pseudoarthrosis, calcaneovalgus deformity, thoracic scoliosis, lumbar hyperlordosis, coxa valga, skeletal dysplasia, pectus carinatum, generalized hypotonia, global developmental delay, cognitive impairment, complete lack of adipose tissue with consequent pseudohypertrophy of muscles, bilateral cataracts and progeroid facial appearance ( and ). As a part of the syndrome, an extremely short stature, body weight of 15 kg and body height of 100 cm were observed.

**Qwen-only** *(FK 8.2)*

A 15-year-old boy had very slow growth in the womb, leading to health issues at birth. At birth, he had unusual facial features like a narrow nose and small hands with short fingers. His legs were also underdeveloped and his spine was curved. He has weak muscles and his body lacks fat.

The CT scan showed an abnormal shape of a bone called the atlas. MRI showed that part of his neck where the skull meets the spine was too narrow for the spinal cord to pass through normally, causing pressure and pain.

Doctors diagnosed this as Seckel Syndrome (SWS), which is a rare disorder characterized by severe short stature along with other health problems. The condition impacts almost every part of the body including the face, bones, and sometimes causes vision complications like cataracts. Despite these challenges, doctors are working on treatments and managing symptoms to improve his quality of life.

**D3 — BioBART→MedJEx→Qwen rewrite** *(FK 25.0)*

A 15-year-old boy suffering from SDS was born to nonconsanguine young and healthy parents from the third pregnancy as a hypotrophic infant with perinatal risks including intrauterine growth retardation and asphyxia. At birth, complex facial and skeletal dysmorphic features were observed in our patient: midface retrusion, prominent forehead, frontal bossing, shallow orbits, downslanted palpebral fissures, narrow nasal bridge, large filtrum, micrognathia, short hands with short metacarpal bones and small fingers with brachydactyly, knee flexion contracture, pseudoarthrosis, calcaneovalgus deformity, thoracic scoliosis, lumbar hyperlordosis, coxa valga, skeletal dysplasia, pectus carinatum, generalized hypotonia, global developmental delay, cognitive impairment, severe lack of adipose tissue with pseudohypertrophy of muscles, bilateral cataracts, and a progeroid facial appearance (and). The syndrome included an extremely short stature (body weight of 15 kg and body height of 100 cm).

### Analysis

| | D2 | Qwen | D3 |
|---|---|---|---|
| **Strengths** | | | |
| **Weaknesses** | | | |
| **Factual accuracy** | | | |
| **Readability** | | | |
| **Completeness** | | | |

**What a better system would do here:**

<!-- fill in -->

---

## Example 3 &nbsp;·&nbsp; ID 3206 &nbsp;·&nbsp; Group A

> High jargon load, large FK gap — D2 much harder to read than Qwen

| Metric | D2 | Qwen | D3 | Gold |
|--------|:--:|:----:|:--:|:----:|
| FK Grade | 17.7 | 7.1 | 14.6 | 15.4 |
| Word count | 190 | 77 | 176 | 52 |
| Source words | 194 | | | |

**MLM jargon spans detected (15 total, top 12 shown):**
`gynecology`, `Zahedan University`, `her vagina and lower abdominal pain`, `Saravan city hospital`, `the child`s vagina`, `frog-leg position`, `tension`, `her vulvae labials major`, `her vagina`, `hymen`, `the hymen`, `child hymen`

<details>
<summary><strong>Source (clinical case report)</strong></summary>

A 4-year-old girl was brought by her mother from Saravan city to the emergency department of gynecology in Ali-Ibn-Abitaleb Hospital of Zahedan University of Medical Sciences in May 2012, with the complaints of a foreign body in her vagina and lower abdominal pain for eight hours. The clinicians did a pelvic X-Ray in Saravan city hospital that showed a big metal nail in the child`s vagina . In physical examination of the girl in frog-leg position and by tension of her vulvae labials major, a black point in depth of her vagina through the orifice of hymen was observed, it was the tail of a nail, the hymen was intact and her mother entreated that we discharge the nail without trauma to child hymen. Fortunately the cooperation of child was very good and we could discharge the nail by doing the TR (rectal exam with finger) and pushing the nail to midline of vaginal canal and extraction of nail out of vagina by use of a magnetic mag . In this case, it could not be established whether the nail had been inserted by the child herself or by another child or an adult.

</details>

<details>
<summary><strong>Gold reference summary</strong></summary>

This problem may produce symptoms or be asymptomatic for long periods of time and may result from ignorance, accident, malice, psychotic tendencies, attempts at sexual stimulation or sexual abuse. The current report presents the case of a girl that had inserted a foreign body in her vagina probably due to childish prank.

</details>

**D2 — BioBART LoRA** *(FK 17.7)*

A 4-year-old girl was brought by her mother to the emergency department of gynecology in Ali-Ibn-Abitaleb Hospital of Zahedan University of Medical Sciences in May 2012, with the complaints of a foreign body in her vagina and lower abdominal pain for eight hours. The clinicians did a pelvic X-Ray in Saravan city hospital that showed a big metal nail in the child`s vagina. In physical examination of the girl in frog-leg position and by tension of her vulvae labials major, a black point in depth of her vagina through the orifice of hymen was observed, it was the tail of a nail, the hymen was intact and her mother entreated that we discharge the nail without trauma to child hymen. Fortunately the cooperation of child was very good and we could discharge this nail by doing the TR (rectal exam with finger) and pushing the nails to midline of vaginal canal and extraction of nail out of vagina by use of a magnetic mag. In this case, it could not be established whether the metal nail had been inserted by the children herself or by another child or an adult.

**Qwen-only** *(FK 7.1)*

A young girl had a metal nail stuck in her vagina and lower stomach pain for 8 hours. An X-ray outside showed the large nail inside. During an exam by doctors, they found only a small part of the nail inside the vagina but could see it's tail sticking out. The doctors carefully removed the nail without hurting her hymen using special tools. They couldn't figure out if she put it there or someone else did accidentally.

**D3 — BioBART→MedJEx→Qwen rewrite** *(FK 14.6)*

A 4-year-old girl was brought by her mother to the emergency department of gynecology in Ali-Ibn-Abitaleb Hospital of Zahedan University of Medical Sciences in May 2012, with complaints of a foreign body in her vagina and lower abdominal pain for eight hours. The X-Ray done at Saravan city hospital showed a big metal nail inside the child's vagina. During physical examination of the girl while she was in the frog-leg position and we pulled on her vulvae labials major, a black point was found deep within the vagina through the hymen orifice—this turned out to be the tail of a nail. The hymen was intact, and her mother requested we remove the nail without causing any trauma to the child's hymen. Fortunately, the child cooperated well, and we were able to remove the nail using the TR (rectal exam) method and a magnet after pushing it towards the midline of the vaginal canal. In this case, it couldn't be determined whether the metal nail had been inserted by the child herself, another child, or an adult.

### Analysis

| | D2 | Qwen | D3 |
|---|---|---|---|
| **Strengths** | | | |
| **Weaknesses** | | | |
| **Factual accuracy** | | | |
| **Readability** | | | |
| **Completeness** | | | |

**What a better system would do here:**

<!-- fill in -->

---

## Example 4 &nbsp;·&nbsp; ID 3269 &nbsp;·&nbsp; Group A

> High jargon load, large FK gap — long source

| Metric | D2 | Qwen | D3 | Gold |
|--------|:--:|:----:|:--:|:----:|
| FK Grade | 19.1 | 9.1 | 15.9 | 19.7 |
| Word count | 206 | 151 | 181 | 147 |
| Source words | 1067 | | | |

**MLM jargon spans detected (23 total, top 12 shown):**
`our center`, `World Federation`, `grade IV`, `diffuse`, `thick SAH`, `a large right temporal intracerebral hematoma`, `a ruptured right middle cerebral artery aneurysm`, `SAH onset`, `aphasia`, `Transcranial Doppler (TCD) measurements`, `subsequent technetium`, `99 m`

<details>
<summary><strong>Source (clinical case report)</strong></summary>

A 63-year-old man admitted to the stroke service at our center presented with the sudden onset of headache, vomiting, and right-sided hemiparesis followed by loss of consciousness (World Federation of Neurological Surgery grade IV). A computed tomography (CT) scan revealed diffuse thick SAH combined with a large right temporal intracerebral hematoma , and a ruptured right middle cerebral artery aneurysm was successfully clipped following extensive hematoma evacuation by emergent surgery performed at 6 hours after SAH onset. The early postoperative course was uneventful, and the neurological deficits disappeared with the exception of mild left hemispatial neglect. He received standard post-SAH fluid and drug management at our institution. On the ninth day after SAH onset, however, he became restless and then developed aphasia and right-sided hemiparesis. Transcranial Doppler (TCD) measurements were compatible with mild left middle cerebral artery (MCA) vasospasm (left MCA peak velocity, 149 cm/s; mean velocity 120 cm/s). Diffusion-weighted magnetic resonance (MR) imaging performed immediately after the onset of symptoms revealed no apparent ischemic findings , but subsequent technetium 99 m hexamethylpropyleneamine oxime single photon emission CT (Tc-99 m HMPAO SPECT) revealed a mild reduction in rCBF in the left anterior cerebral artery (ACA) and MCA territories ] indicative of clinical deterioration attributable to vasospasm. Hyperdynamic therapy with dobutamine (initial dose: 3 μg/kg/min; increased in 3 μg/kg/min increments every 4 hours to a level at which the deterioration resolved) combined with mild hypervolemia with supplemental low-molecular-weight dextran (500 mL/day) was initiated. Cerebral digital subtraction angiography (DSA) was performed under

*[source truncated at 250 words — full length: 1067 words]*

</details>

<details>
<summary><strong>Gold reference summary</strong></summary>

A 63-year-old man presented with SAH and intracerebral hematoma due to ruptured right middle cerebral artery aneurysm developed aphasia and right-sided weakness on day 9 after SAH onset. Delayed cerebral ischemia attributable to diffuse vasospasm in the distal territories of the left anterior and middle cerebral arteries was suspected. Since the symptoms persisted despite maximal hyperdynamic therapy with dobutamine, intra-arterial fasudil treatment in the setting of rSO(2) monitoring including the spasm-affected vascular territory with four-channel flexible NIRS sensors was subsequently performed. Decreased and fluctuating rSO(2) in angiographically documented vasospastic territories increased immediately after intra-arterial fasudil infusion in accordance with relief of vasospasm that correlated with neurological improvement. The procedure was repeated on day 11 since the effect was transient and neurological deterioration and reduction of rSO(2) recurred. The deficits resolved accompanied by uptake and maintenance of rSO (2) following the intra-arterial fasudil, resulting in favorable functional outcome.

</details>

**D2 — BioBART LoRA** *(FK 19.1)*

A 63-year-old man admitted to the stroke service at our center presented with the sudden onset of headache, vomiting, and right-sided hemiparesis followed by loss of consciousness (World Federation of Neurological Surgery grade IV). A computed tomography (CT) scan revealed diffuse thick SAH combined with a large right temporal intracerebral hematoma, and a ruptured right middle cerebral artery aneurysm was successfully clipped. After surgery, the patient's neurological deficits gradually improved. However, on the ninth day after SAH onset, he became restless and then developed aphasia. Transcranial Doppler (TCD) measurements were compatible with mild left Middle cerebral artery (MCA) vasospasm (left MCA peak velocity, 149 cm/s; mean velocity 120 cm/ s). Diffusion-weighted magnetic resonance (MR) imaging performed immediately after the onset of symptoms revealed no apparent ischemic findings, but subsequent technetium 99 m hexamethylpropyleneamine oxime single photon emission CT (Tc-99 m HMPAO SPECT) revealed a mild reduction in rCBF in the left anterior cerebral artery and MCA territories ] indicative of clinical deterioration attributable to vasospasm. Hyperdynamic therapy with dobutamine (initial dose: 3 μg/kg/min; increased in 3 μg × 3 μg /kg/ min increments every 4 hours to a level at which the deterioration resolved) combined with mild hypervolemia with supplemental low-molecular-weight dextran (500 mL/day) was initiated.

**Qwen-only** *(FK 9.1)*

A 63-year-old man suddenly developed a headache, vomiting, and right-sided weakness followed by unconsciousness. CT showed bleeding in his brain and an untreated ruptured aneurysm which was clipped surgically. He remained alert following surgery but became restless with speech problems and weakness on the right side nine days later. Tests indicated that the problem was likely vasospasm in certain blood vessels. Doctors tried giving him a special drug to dilate the arteries, but it stopped working two hours later. A second treatment session was done under local anesthesia. They used a tiny device to infuse a drug called fasudil into specific areas of his brain. This helped improve blood flow and symptoms quickly. Despite this, he had some persistent issues that took several more days to fully resolve. However, the treatment prevented further complications or major new problems. After a ventricle shunt procedure to treat hydrocephalus, his function is improving well.

**D3 — BioBART→MedJEx→Qwen rewrite** *(FK 15.9)*

A 63-year-old man admitted to our center presented with sudden headache, vomiting, and right-sided weakness followed by loss of consciousness (World Federation of Neurological Surgery grade IV). A CT scan showed a thick subarachnoid hemorrhage (SAH) spread throughout the brain, along with a large hematoma in the right temporal lobe. An aneurysm on the right middle cerebral artery was successfully clipped.

After surgery, his neurological symptoms improved but worsened again nine days after the SAH onset. He became agitated and developed aphasia. Transcranial Doppler measurements suggested mild left MCA vasospasm (left MCA peak velocity, 149 cm/s; mean velocity: 120 cm/s).

Diffusion-weighted magnetic resonance imaging did not show significant ischemia at the onset of symptoms, but a technetium 99 m hexamethylpropyleneamine oxime single photon emission computed tomography (Tc-99 m HMPAO SPECT) revealed reduced blood flow in the left anterior cerebral artery and MCA territories, indicating clinical deterioration due to vasospasm.

Hyperdynamic therapy started with dobutamine (initial dose: 3 μg/kg/min; increased by 3 μg/μg kg/h every 4 hours until the condition stabilized), along with mild hypervolemia treated with supplemental low-molecular-weight dextran (500 mL/day).

### Analysis

| | D2 | Qwen | D3 |
|---|---|---|---|
| **Strengths** | | | |
| **Weaknesses** | | | |
| **Factual accuracy** | | | |
| **Readability** | | | |
| **Completeness** | | | |

**What a better system would do here:**

<!-- fill in -->

---

## Example 5 &nbsp;·&nbsp; ID 1306 &nbsp;·&nbsp; Group A

> High jargon load, large FK gap — D2 much harder to read than Qwen

| Metric | D2 | Qwen | D3 | Gold |
|--------|:--:|:----:|:--:|:----:|
| FK Grade | 17.4 | 7.5 | 17.8 | 14.8 |
| Word count | 182 | 183 | 176 | 84 |
| Source words | 250 | | | |

**MLM jargon spans detected (23 total, top 12 shown):**
`an unremarkable medical history`, `an asymptomatic retinal hemorrhage`, `laser photocoagulation`, `Optical coherence tomography`, `angiographic flow overlay`, `a peaked PED`, `Ophthalmoscopic examination`, `subretinal hemorrhage`, `a pigment epithelial detachment`, `the superotemporal vascular arcade`, `Spectral-domain optical coherence tography`, `Multiple, small, cuticular drusen`

<details>
<summary><strong>Source (clinical case report)</strong></summary>

A 61-year-old white woman with an unremarkable medical history was referred for an evaluation of an asymptomatic retinal hemorrhage detected in her right eye. Her past ocular history was significant for a complete posterior vitreous detachment in the right eye. Her best-corrected visual acuity was 20/20 in each eye. Anterior segment examination and applanation tensions were unremarkable. Ophthalmoscopic examination of the right eye identified subretinal hemorrhage surrounding a pigment epithelial detachment (PED) located above the superotemporal vascular arcade . Spectral-domain optical coherence tomography (SD-OCT) demonstrated the presence of a PED accompanied by irregularities of the retinal pigment epithelium (RPE) profile . Optical coherence tomography (OCT) B-scan with angiographic flow overlay showed a peaked PED with intrinsic flow signal . Ophthalmoscopic examination of the left eye demonstrated a PED nasal to the optic nerve characterized by a dome-shaped elevation of the RPE with a shallow irregular portion on SD-OCT , whereas the angiographic flow overlay revealed intrinsic flow signal . Multiple, small, cuticular drusen appearing as small hypoautofluorescent dots on fundus autofluorescence (FAF) were seen in both eyes along the vascular arcades . Indocyanine green angiography (ICGA) showed focal areas of hyperfluorescence within the PEDs, indicative of AT1 .
OCT angiography en face slab demonstrated aneurysmal dilatation arising from a type 1 neovascular network, particularly evident in the right eye with active lesions. As comparative imaging from the patient’s prior examination showed increased hemorrhage, treatment with anti-vascular endothelial growth factor (VEGF) therapy was initiated for the right eye at that time.

</details>

<details>
<summary><strong>Gold reference summary</strong></summary>

A 61-year-old woman of European descent was referred for a new, asymptomatic retinal hemorrhage found on routine examination. Ophthalmoscopy revealed cuticular drusen in both eyes best appreciated on fundus autofluorescence, and a hemorrhagic retinal pigment epithelium detachment above the superior arcade in the right eye. In the fellow eye, a reddish appearing pigment epithelial detachment was noted nasal to the optic nerve. Indocyanine green angiography showed findings of AT1 in both eyes. Optical coherence tomography angiography showed intrinsic flow signal within the aneurysmal lesions.

</details>

**D2 — BioBART LoRA** *(FK 17.4)*

A 61-year-old white woman with an unremarkable medical history was referred for an evaluation of an asymptomatic retinal hemorrhage detected in her right eye. Her past ocular history was significant for a complete posterior vitreous detachment in the right eye, which was treated with laser photocoagulation. Optical coherence tomography (OCT) B-scan with angiographic flow overlay showed a peaked PED with intrinsic flow signal. Ophthalmoscopic examination of the left eye demonstrated subretinal hemorrhage surrounding a pigment epithelial detachment (PED) located above the superotemporal vascular arcade. Spectral-domain optical coherence tography revealed the presence of a PED with a shallow irregular portion. Multiple, small, cuticular drusen appearing as small hypoautofluorescent dots on fundus autofluorescence (FAF) were seen in both eyes along the vascular arcades. Indocyanine green angiography showed focal areas of hyperfluorescence within the PEDs, indicative of AT1. Unloaded-OCT angiographies en face slab demonstrated aneurysmal dilatation arising from a type 1 neovascular network, particularly evident in that eye with active lesions. As comparative imaging from the patient’s prior examination showed increased hemorrhage, treatment with anti-vascular endothelial growth factor (VEGF) therapy was initiated for that time.

**Qwen-only** *(FK 7.5)*

A 61-year-old woman noticed a spot in her right eye that didn't cause any pain or vision trouble. An eye doctor found red blood cells under the retina in one eye during a routine check-up. The other eye had a small part of its back torn but was seeing normally. When the doctor looked closely at both eyes, they saw a problem called a pigment epithelial detachment (PED) above where veins are located, especially on the upper outside of the eye. Using special imaging, they also saw a type of leaky vein called an AT1 lesion and signs of bleeding inside the PED.

In tests of the left eye, there was a smaller version of this same problem near the optic nerve. The back of both eyes have small spots called cuticular drusen which could be seen with a special light. Another test showed some areas of the leaks were active.

To treat the right eye, the doctor put medicine into it to stop new blood vessels from forming. Now, we will follow up to see if the treatment helped reduce the bleeding.

**D3 — BioBART→MedJEx→Qwen rewrite** *(FK 17.8)*

A 61-year-old white woman with an unremarkable medical history was referred for an evaluation of an asymptomatic retinal hemorrhage in her right eye. Her past ocular history included a complete posterior vitreous detachment in the right eye that was treated with laser photocoagulation. Optical coherence tomography (OCT) B-scan with angiographic flow overlay showed a peaked PED with intrinsic flow signal. An internal opthalmoscopic examination of the left eye revealed subretinal hemorrhage surrounding a pigment epithelial detachment (PED) located above the superotemporal vascular arcade. Spectral-domain optical coherence tography indicated the presence of a PED with a shallow irregular portion. Small cuticular drusen appearing as small hypoautofluorescent dots were seen on fundus autofluorescence (FAF) in both eyes, particularly along the vascular arcades. Indocyanine green angiography displayed focal areas of hyperfluorescence within the PEDs, suggesting atypical type 1 neovascularization. Unloaded-OCT angiographies en face slab demonstrated aneurysmal dilatation arising from a type 1 neovascular network, primarily evident in her active lesions. Comparative imaging from her prior examination showed increased hemorrhage, leading to treatment with anti-vascular endothelial growth factor (VEGF) therapy.

### Analysis

| | D2 | Qwen | D3 |
|---|---|---|---|
| **Strengths** | | | |
| **Weaknesses** | | | |
| **Factual accuracy** | | | |
| **Readability** | | | |
| **Completeness** | | | |

**What a better system would do here:**

<!-- fill in -->

---

## Example 6 &nbsp;·&nbsp; ID 2272 &nbsp;·&nbsp; Group A

> High jargon load, large FK gap — D2 much harder to read than Qwen

| Metric | D2 | Qwen | D3 | Gold |
|--------|:--:|:----:|:--:|:----:|
| FK Grade | 17.5 | 7.8 | 17.5 | 17.8 |
| Word count | 129 | 141 | 128 | 69 |
| Source words | 435 | | | |

**MLM jargon spans detected (19 total, top 12 shown):**
`A 74-year-old male patient`, `30% stenosis`, `the RCA, atrial fibrillation`, `chronic anticoagulation`, `amiodarone`, `diltiazem`, `implantable defibrillator`, `self-resolved upper respiratory infection`, `orthopnea`, `paroxysmal nocturnal dyspnea`, `diaphoresis`, `self-resolving`

<details>
<summary><strong>Source (clinical case report)</strong></summary>

A 74-year-old male patient with chronic kidney disease stage III, essential hypertension, coronary artery disease with 30% stenosis of the RCA, atrial fibrillation on chronic anticoagulation, rhythm control with amiodarone, rate control with diltiazem, and implantable defibrillator, who had a history of self-resolved upper respiratory infection a week prior to admission, when he started experiencing shortness of breath (SOB). Shortness of breath was evident during exertion and at rest, presenting with orthopnea and paroxysmal nocturnal dyspnea. His symptoms were accompanied by intermittent chest discomfort and pressure lasting around 15 minutes, worse when laying supine, attenuated by sitting up, non-radiating, and not accompanied by diaphoresis, and self-resolving. He did not take any medications to alleviate discomfort. He mentioned experiencing chills, diaphoresis, nasal congestion, sore throat, arthralgia, unexpected weight loss, and occasional dizziness and imbalance while walking. The day before admission, he could walk his dog for 1 block; on admission day, he could not walk due to persistent SOB. He denied having any fever, cough, hemoptysis, palpitations, edema, muscle, or joint discomfort. On physical examination his vital signs demonstrated heart rate 74 beats per minute, blood pressure 159/69 mmHg, respiratory rate 20, O2 sat above 94% on room air. He was in respiratory distress and his cardiovascular assessment revealed distant heart sounds, absence of JVD, abdominal tenderness in the right upper quadrant and epigastric area, as well as malleolar pitting edema. Electrocardiogram demonstrated sinus rhythm, 70 bpm, prolonged PR, low voltage in precordial leads, and prolonged QTc. Findings such as muffled

*[source truncated at 250 words — full length: 435 words]*

</details>

<details>
<summary><strong>Gold reference summary</strong></summary>

We present the case of a 74-year-old male with a history of upper respiratory infection who developed sudden onset dyspnea and chest discomfort. Bedside point-of-care ultrasound (POCUS) revealed a large pericardial effusion, prompting urgent intervention. Despite initially stable vital signs, the patient rapidly deteriorated, necessitating emergent pericardiocentesis. Laboratory findings and pathology results eventually ruled out common viral causes, guiding diagnosis toward coxsackieviruses A and B, echovirus, adenoviruses, or influenza.

</details>

**D2 — BioBART LoRA** *(FK 17.5)*

A 74-year-old male patient with chronic kidney disease stage III, essential hypertension, coronary artery disease with 30% stenosis of the RCA, atrial fibrillation on chronic anticoagulation, rhythm control with amiodarone, rate control with diltiazem, and implantable defibrillator, who had a history of self-resolved upper respiratory infection a week prior to admission, presented with orthopnea and paroxysmal nocturnal dyspnea. His symptoms were accompanied by intermittent chest discomfort and pressure lasting around 15 minutes, attenuated by sitting up, non-radiating, and not accompanied by diaphoresis, and self-resolving. He did not take any medications to alleviate discomfort. Chest x-ray demonstrated cardiomegaly. Pericardial window was performed, draining the remaining pericardial effusion (700 mL of bloody, nonclotting fluid); the pericardium and epicardium were extremely inflamed. No isolated source of bleeding; however, some areas of oozing.

**Qwen-only** *(FK 7.8)*

A 74-year-old man with heart and kidney problems was admitted due to shortness of breath, chest discomfort, and fainting spells. He couldn’t walk far after a week of feeling sick. Doctors heard his heartbeat was weak through his chest and found fluid around his heart using an imaging test.

When he came in, he got very weak suddenly. They did emergency surgery to drain the heart and take out some sticky fluid that was causing pressure on his heart.

Tests showed there was no infection or broken bone as a reason for the trouble. The fluid they removed wasn’t from cancer but had lots of fat cells, suggesting it might be from normal healing processes going too far.

Doctors treated him at once with more fluids and medicines to keep his blood pressure up while waiting for his condition to improve.

**D3 — BioBART→MedJEx→Qwen rewrite** *(FK 17.5)*

A 74-year-old male patient with chronic kidney disease stage III, essential hypertension, coronary artery disease with 30% stenosis of the RCA, atrial fibrillation on chronic anticoagulation, rhythm control with amiodarone, rate control with diltiazem, and implantable defibrillator, who had a history of self-resolved upper respiratory infection a week prior to admission, presented with orthopnea and paroxysmal nocturnal dyspnea. His symptoms were accompanied by intermittent chest discomfort and pressure lasting around 15 minutes, attenuated by sitting up, not radiating elsewhere, and not accompanied by diaphoresis, and would resolve on its own. Chest x-ray showed cardiomegaly. A pericardial window was performed, draining the remaining pericardial effusion (700 mL of nonclotting, bloody fluid); the pericardium and epicardium were extremely inflamed. There was no isolated source of bleeding, but some areas oozed.

### Analysis

| | D2 | Qwen | D3 |
|---|---|---|---|
| **Strengths** | | | |
| **Weaknesses** | | | |
| **Factual accuracy** | | | |
| **Readability** | | | |
| **Completeness** | | | |

**What a better system would do here:**

<!-- fill in -->

---

## Example 7 &nbsp;·&nbsp; ID 1784 &nbsp;·&nbsp; Group B

> D3 achieves lowest FK — rewrite working well

| Metric | D2 | Qwen | D3 | Gold |
|--------|:--:|:----:|:--:|:----:|
| FK Grade | 12.2 | 18.2 | 6.0 | 13.6 |
| Word count | 214 | 128 | 202 | 144 |
| Source words | 812 | | | |

**MLM jargon spans detected (16 total, top 12 shown):**
`urinary hesitancy`, `urination`, `291.0 μmol`, `Obstructive nephropathy`, `renal dysfunction`, `renal division`, `anti-neutrophil cytoplasmic antibodies`, `The abdominal ultrasonography`, `multiple solid nodules`, `multiple liver parenchymal round`, `gastrointestinal symptoms`, `positive fecal occult blood test`

<details>
<summary><strong>Source (clinical case report)</strong></summary>

A 71-year-old Chinese man presented with urinary hesitancy, dribbling urination, and prolonged urination and was diagnosed as benign prostatic hyperplasia at out-patient one year ago. The serum creatinine was 101 μmol/L (normal range 53~140 μmol/L) at that moment. He was prescribed with epristeride and tamsulosin. Nine months ago, the patient stopped the oral medication because of loss of appetite. The symptoms of urinary hesitancy, dribbling and prolonged urination worsened gradually and therefore he was admitted to our hospital for surgery. On admission, the renal function test revealed a serum creatinine level of 291.0 μmol/L. The post-void residual was normal. The ultrasonic examination revealed that both kidneys were normal in structure and size (left 11.6 cm × 6.3 cm,right 10.7 cm × 4.4 cm). Obstructive nephropathy was thus excluded and the surgery was canceled for renal dysfunction. The patient was transferred to renal division of internal medicine department where additional tests were performed in order to establish the etiology of his documented renal failure. The results of routine peripheral blood test were as follows: hemoglobin 89 g/L (normal range 130~175 g/L), white blood cells 5.21 × 109/L (normal range 3.5~9.55.21 × 109/L), and platelets 204 × 109/L (normal range 100~300 × 109/L). Urinalysis was positive for 1+ protein. Red blood cells and white blood cells were negative in urine sediment microscopic examination. The 24 h urinary protein determination was 0.67 g. Fecal occult blood testing was positive. In addition, the serum creatinine level increased to 415 μmol/L. The immunology tests revealed

*[source truncated at 250 words — full length: 812 words]*

</details>

<details>
<summary><strong>Gold reference summary</strong></summary>

A 71-year-old Chinese man presented with dysuria and was initially diagnosed as benign prostatic hyperplasia for one year. He was admitted to the hospital for surgery. After admission, the renal function tests revealed a rapid increase of serum creatinine from 291.0 μmol/L to 415 μmol/L. The hemoglobin level was 89 g/L. Fecal occult blood testing was positive. Urinalysis revealed mild proteinuria. The serum IgG4 level was 13.9 g/L. The abdominal imaging examination revealed multiple solid nodules in the liver. The gastrointestinal endoscopy combined with the biopsy revealed colon adenocarcinoma. Kidney biopsy showed massive IgG4-positive plasma cells and storiform fibrosis infiltration in the tubulointerstitial area, thus establishing the diagnosis of IgG4-related tubulointerstitial nephritis. Corticosteroid therapy was initiated, and subsequently, the renal function dramatically improved without the diminution of the liver nodules. The liver biopsy was performed and a diagnosis of metastatic colon adenocarcinoma was confirmed.

</details>

**D2 — BioBART LoRA** *(FK 12.2)*

A 71-year-old Chinese man presented with urinary hesitancy, dribbling urination, and prolonged urination. The renal function test revealed a serum creatinine level of 291.0 μmol/L. The post-void residual was normal. The ultrasonic examination revealed that both kidneys were normal in structure and size (left 11.6 cm × 6.3 cm,right 10.7 cm × 4.4 cm). Obstructive nephropathy was thus excluded and the surgery was canceled for renal dysfunction. The patient was transferred to renal division of internal medicine department where additional tests were performed in order to establish the etiology of his documented renal failure. The immunology tests revealed the following: anti-neutrophil cytoplasmic antibodies and anti-glomerular basement membrane antibody were both negative. The abdominal ultrasonography revealed multiple solid nodules in the liver. Magnetic resonance imaging (MRI) confirmed multiple liver parenchymal round shaped long T1 and long T2 signal nodules, with a diameter of between 0.6 and 16 cm. The nodules revealed mild enhancement during arterial enhancement phase with some of them showed a decline of enhancement at portal enhancement period. Since the patient has gastrointestinal symptoms in combination with positive fecal occult blood test and moderate anemia, a gastrointestinal endoscopy was performed and It showed a circular cauliflower shaped, ulcerative mass at the middle section of the transverse colon. Biopsy of this mass revealed adenocarcinoma.

**Qwen-only** *(FK 18.2)*

A 71-year-old man with benign prostatic hyperplasia presented with worsening urinary symptoms, stopping medication and worsening renal function. His serum creatinine was elevated at 291 μmol/L when he was admitted for surgery to exclude obstructive nephropathy. Despite initial surgery cancellation due to renal dysfunction, further tests identified multiple liver nodules and signs of systemic immune response, leading to a diagnosis of IgG4-related tubulointerstitial nephritis (IgG4-TIN). Prednisone was initiated, initially improving his kidney function but without significant reduction in the liver nodules on repeat MRI.

Prednisone continued while preparing the patient for chemotherapy for colon adenocarcinoma, which also had advanced metastasis to the liver. However, five months later, the patient’s general condition deteriorated and he passed away unexpectedly, dying from respiratory failure despite improved kidney function from prednisone therapy.

**D3 — BioBART→MedJEx→Qwen rewrite** *(FK 6.0)*

A 71-year-old Chinese man came with difficulty starting urination, dribbling urine, and a delay in urination. His kidney function showed a serum creatinine level of 291.0 μmol/L. The amount of urine left in his bladder after trying to void was normal. Ultrasound checked both kidneys; they looked fine but were small (left kidney: 11.6 cm × 6.3 cm, right kidney: 10.7 cm × 4.4 cm). Since the obstruction wasn't found and there was a problem with how well his kidneys work, surgery wasn't done. He was sent to the kidney division of medicine where doctors did more tests to find what was causing his kidneys not to work well. Lab tests showed no signs of inflammation in the blood that might be attacking his kidneys. An ultrasound also found some round lumps in the liver. MRI showed similar lumps with a diameter ranging from 0.6 to 16 cm. Some had changes after being fed blood during special scans. Because the patient was having problems with his gut and his poop test showed he has a mass, doctors asked him to do an endoscopy where they saw a big red-looking growth about halfway down his small intestine. Biopsy of this growth showed cancer.

### Analysis

| | D2 | Qwen | D3 |
|---|---|---|---|
| **Strengths** | | | |
| **Weaknesses** | | | |
| **Factual accuracy** | | | |
| **Readability** | | | |
| **Completeness** | | | |

**What a better system would do here:**

<!-- fill in -->

---

## Example 8 &nbsp;·&nbsp; ID 515 &nbsp;·&nbsp; Group B

> D3 achieves lowest FK — rewrite working well

| Metric | D2 | Qwen | D3 | Gold |
|--------|:--:|:----:|:--:|:----:|
| FK Grade | 9.0 | 10.7 | 6.1 | 15.1 |
| Word count | 131 | 145 | 135 | 86 |
| Source words | 426 | | | |

**MLM jargon spans detected (12 total, top 12 shown):**
`productive cough`, `progressive dyspnea`, `exertion`, `only osteomyelitis`, `dust`, `an occupational hazard`, `SpO2 level`, `room air`, `bilateral ground glass opacities`, `thickened interlobular septa`, `an appearance`, `the ‘crazy-paving’ pattern`

<details>
<summary><strong>Source (clinical case report)</strong></summary>

A 72-year-old Japanese woman presented to our hospital with a one-year history of productive cough and progressive dyspnea on exertion. Her past medical history included only osteomyelitis at 35-years-old and she was not taking any medications. She had never smoked or inhaled dust as an occupational hazard. Her vital signs were as follows: body temperature of 36.6°C, blood pressure of 130/80mmHg, heart rate of 67 beats/min, and SpO2 level of 80% (room air). Her arterial blood gas analysis revealed a pH of 7.44, PaO2 level of 43.2mmHg, PaCO2 level of 39.2mmHg, and HCO3− level of 26.4mEq/L on room air. Her chest radiographs (CXR) showed bilateral infiltrates in her mid and lower lung zones , and her chest CT scan showed bilateral ground glass opacities with thickened interlobular septa, an appearance known as the ‘crazy-paving’ pattern . Her laboratory tests revealed a normal complete blood count, high lactate dehydrogenase (543/mL), and high Krebs von den Lungen-6 level (16,189U/mL). Her pulmonary function tests taken on admission revealed a vital capacity (VC) of 1.49L, %VC of 68.6%, forced expiratory volume in one second (FEV1) of 1.17L, FEV1/forced vital capacity (FVC) predicted ratio of 76.0%, and FEV1/FVC actual ratio of 84.2%. We retrieved 26/100mL of bronchoalveolar lavage fluid (BALF), which was milky in appearance. Her BALF total cell count was 6.0×104/mL; cell differentiation revealed 23% neutrophils, 1% eosinophils, 17% lymphocytes, and 59% macrophages. Her BALF contained copious eosinophilic, periodic acid Schiff-positive granular material, a finding that supported a diagnosis of PAP. In addition, a

*[source truncated at 250 words — full length: 426 words]*

</details>

<details>
<summary><strong>Gold reference summary</strong></summary>

A 72-year-old Japanese woman presented to our hospital with a one-year history of productive cough and progressive dyspnea. Her chest computed tomography scan showed a bilateral crazy-paving pattern in both of her lungs. She was diagnosed with autoimmune pulmonary alveolar proteinosis based on bronchoalveolar lavage findings and the presence of serum anti-granulocyte macrophage colony-stimulating factor antibodies. She was severely hypoxemic, so we recommended whole lung lavage or inhaled granulocyte macrophage colony-stimulating factor treatment, which she refused. We initiated treatment with ambroxol and her symptoms markedly improved.

</details>

**D2 — BioBART LoRA** *(FK 9.0)*

A 72-year-old Japanese woman presented to our hospital with a one-year history of productive cough and progressive dyspnea on exertion. Her past medical history included only osteomyelitis at 35-years-old and she was not taking any medications. She had never smoked or inhaled dust as an occupational hazard. Her vital signs were as follows: body temperature of 36.6°C, blood pressure of 130/80mmHg, heart rate of 67 beats/min, and SpO2 level of 80% (room air). Her arterial blood gas analysis revealed a pH of 7.44, PaO2 level, 43.2mmHg, PaCO2 level and HCO3− level of 26.4mEq/L on room air. Her chest radiographs (CXR) showed bilateral infiltrates in her mid and lower lung zones, and her chest CT scan showed bilateral ground glass opacities with thickened interlobular septa, an appearance known as the ‘crazy-paving’ pattern.

**Qwen-only** *(FK 10.7)*

A 72-year-old Japanese woman came to the hospital with a persistent productive cough and worsening shortness of breath. On examination, she had low oxygen levels and abnormal lung x-rays showing fluid in her lungs. Blood and lung fluid tests revealed high levels of certain inflammatory markers and an unusual material called PAP was present in her lungs due to an autoimmune condition.

Doctors diagnosed her with very severe PAP and treated her with daily medication including ambroxol. She was not admitted to the hospital but given oxygen at home. She reported that loosening her mucus helped improve her cough and breathing as her oxygen levels rose to normal. Over time, CT scans showed fewer abnormalities in her lungs, and pulmonary function improved greatly. 

She continues on ambroxol for nearly two years without any worsening symptoms. Treatment helps her feel much better and breathe more easily.

**D3 — BioBART→MedJEx→Qwen rewrite** *(FK 6.1)*

A 72-year-old Japanese woman came to our hospital with a one-year history of productive cough and progressive shortness of breath when exerting herself. She had only osteomyelitis at age 35 and didn't take any medications. She had never smoked or been exposed to dust in her job. Her vital signs were: temperature 36.6°C, blood pressure 130/80 mmHg, heart rate 67 beats per minute, and SpO2 level 80% on room air. Her arterial blood gas showed pH 7.44, PaO2 43.2 mmHg, and PaCO2 and HCO3− levels of 26.4 mEq/L on room air. Chest X-rays (CXR) showed grayish shadows over the middle and lower lung zones. A chest CT scan revealed both lungs had fuzzy-looking areas filled with fluid called ground glass opacities, with thickened lines between the lobes of the lungs, appearing like a crazy-paving pattern.

### Analysis

| | D2 | Qwen | D3 |
|---|---|---|---|
| **Strengths** | | | |
| **Weaknesses** | | | |
| **Factual accuracy** | | | |
| **Readability** | | | |
| **Completeness** | | | |

**What a better system would do here:**

<!-- fill in -->

---

## Example 9 &nbsp;·&nbsp; ID 1594 &nbsp;·&nbsp; Group B

> D3 achieves lowest FK — rewrite working well

| Metric | D2 | Qwen | D3 | Gold |
|--------|:--:|:----:|:--:|:----:|
| FK Grade | 9.2 | 9.7 | 6.2 | 13.1 |
| Word count | 135 | 252 | 184 | 67 |
| Source words | 292 | | | |

**MLM jargon spans detected (18 total, top 12 shown):**
`A 34-year old multigravida`, `right adnexal mass`, `her routine gynecologic examination`, `Pap smear`, `Transvaginal ultrasonography`, `laparoscopy`, `a dilated fallopian tube`, `bluish discoloration`, `the abdomino-pelvic cavity`, `smooth and shiny peritoneal surphace`, `The fimbriae`, `Fine needle aspiration`

<details>
<summary><strong>Source (clinical case report)</strong></summary>

A 34-year old multigravida was found to have right adnexal mass on her routine gynecologic examination. Her previous medical history was uneventfull and Pap smear was normal. Transvaginal ultrasonography identified a cystic mass adjacent to the right ovary. Serum CA 125 was 5.1 U/ml (reference range: < 35 U/ml). At laparoscopy a dilated fallopian tube with bluish discoloration was found. The contralateral fallopian tube, ovaries and uterus were unremarkable. Exploration of the abdomino-pelvic cavity revealed smooth and shiny peritoneal surphace. Obtained peritoneal and pelvic washing were negative. Fine needle aspiration of dilated part of the fallopian tube revealed a 4 ml of bloody content. Cytological findings were consistent with hematosalpinx. Right salpingectomy was performed without using endoscopic bag. The patient was followed up by means of ultrasonography and serum CA 125 for 4.6 years. During this period she had no evidence of the disease. Grossly, a 7.0 cm long fallopian tube was irregularly dilated up to 4.4 cm in diameter at the ampulary region. The fimbriae were intact. The serosal surphace was smooth. Sections of the dilated part of the fallopian tube revealed a cystic tumor with focally yellow to tan, soft papillary excrescences protruding into the lumen and foci of intracystic hemorrhage. On microscopic examination the papillae were covered by serous type of epithelium, displaying stratification and budding with focal nuclear atypia . Three types of cells were recognised; ciliated cells, hob-nail cells and mesothelium-like cells. Small foci of tumor tissue necroses and hemorrhage were noted. There was no

*[source truncated at 250 words — full length: 292 words]*

</details>

<details>
<summary><strong>Gold reference summary</strong></summary>

A case of serous borderline tumor of the fallopian tube in a 34-year old patient is presented, incidentally found during routine gynecologic examination. At laparoscopy the tumor was unusually presented as hematosalpinx and was treated by salpingectomy. Cell-cycle analysis of the tumor tissue revealed a diploid DNA content and a low S-phase fraction. There was no evidence of the disease during the follow-up period of 4.6 years.

</details>

**D2 — BioBART LoRA** *(FK 9.2)*

A 34-year old multigravida was found to have right adnexal mass on her routine gynecologic examination. Her previous medical history was uneventfull and Pap smear was normal. Transvaginal ultrasonography identified a cystic mass adjacent to the right ovary. Serum CA 125 was 5.1 U/ml (reference range: < 35 U/ ml). At laparoscopy a dilated fallopian tube with bluish discoloration was found. Both ovaries were normal. Exploration of the abdomino-pelvic cavity revealed smooth and shiny peritoneal surphace. The fimbriae were intact. Fine needle aspiration (FNA) of dilated part of tube revealed a 4 ml of bloody content. Cytological findings were consistent with hematosalpinx. Right salpingectomy was performed without using endoscopic bag. The patient was followed up by means of sonography and serum CA 125 for 4.6 years. During this period she had no evidence of tumor.

**Qwen-only** *(FK 9.7)*

A 34-year-old woman with multiple pregnancies had a discovered cystic mass near her right ovary during a check-up. Her Pap test was normal and lab blood levels were within the typical range except for slightly elevated CA125, a marker sometimes used for cancers. A closer look at the fallopian tube using cameras inside her body showed it to be partially blocked with dark red liquid inside. The doctors took tissue samples from this part of the tube which showed bleeding and an unusual growth.

Surgery removed just the blockage on her right side. For monitoring over 4 years after, she saw nothing wrong. When looked at closely, the abnormal fallopian tube was very twisted like sausage but its inner lining wasn't damaged or spreading into surrounding tissues. Under the microscope, some abnormal cells grew outwards into the tube's interior. These could form fluid-filled bumps called papillae that had small areas of dead tissue mixed in. Most of these cancer-like changes seemed confined, without invading further.

Genetic testing found there were normal numbers of cell copies making up most of the tumor but they tended to divide slower than average normal cells. This suggested slow growing disease. After all tests, the overall diagnosis was that she had a non-invasive cervical canal tumor within her fallopian tube. Because only one tube was affected and no other signs, doctors felt she did not have a more widespread type. Follow-ups continue to watch for new issues as she may carry risk if cancer happens anywhere else.

**D3 — BioBART→MedJEx→Qwen rewrite** *(FK 6.2)*

A 34-year-old woman who had given birth more than once was found to have a mass on her routine gynecological checkup. Her health history before that was without any issues, and the Pap smear results were normal. A transvaginal ultrasound showed a cyst near her right ovary. Her CA 125 level was 5.1 U/mL (normal range: < 35 U/ML). During laparoscopy, they saw a dilated fallopian tube with a bluish color. Both ovaries looked fine. The doctor checked the area inside the tummy and pelvis, and it felt smooth and shiny. The ends of the fallopian tubes seemed okay. They took a sample from the swollen part of the tube using a thin needle, which came back as bloody fluid. It matched the findings of having blood in the fallopian tube after looking at the cells under a microscope. They removed the right fallopian tube. She wasn't told to use an endoscopic bag during the surgery. For the next 4.6 years, she had regular follow-ups with ultrasounds and checking her CA 125 levels. Over this time, there was no sign of any tumor growth.

### Analysis

| | D2 | Qwen | D3 |
|---|---|---|---|
| **Strengths** | | | |
| **Weaknesses** | | | |
| **Factual accuracy** | | | |
| **Readability** | | | |
| **Completeness** | | | |

**What a better system would do here:**

<!-- fill in -->

---

## Example 10 &nbsp;·&nbsp; ID 3389 &nbsp;·&nbsp; Group B

> D3 achieves lowest FK — rewrite working well

| Metric | D2 | Qwen | D3 | Gold |
|--------|:--:|:----:|:--:|:----:|
| FK Grade | 10.2 | 10.0 | 6.2 | 16.2 |
| Word count | 104 | 134 | 111 | 68 |
| Source words | 402 | | | |

**MLM jargon spans detected (10 total, top 10 shown):**
`stomach`, `spinal surgical intervention`, `dislocation`, `thoracolumbar pedicle`, `‘freehand technique`, `fluoroscopic guidance`, `Eight pedicle screws`, `contoured rods`, `deformities`, `The postoperative course`

<details>
<summary><strong>Source (clinical case report)</strong></summary>

A 21-year-old man presented to the emergency room (ER) complaining of stomach and serious back pain 2 months after spinal surgical intervention. Two months before his visit, he fell from a 10 m tall building. He was confirmed as suffering fracture and dislocation from T11 to L2 by a computed tomography (CT) examination. The patient complained of severe back pain. There was no neural deficit. The patient underwent thoracolumbar pedicle screw implant surgery using ‘freehand technique’ under fluoroscopic guidance in a local hospital. Eight pedicle screws with contoured rods were placed spanning T11 to L2 after the correction of deformities. The patient tolerated the procedure well and developed no neurodeficits postoperatively. Radiological examination revealed a partial correction of the deformity. The patient was presented with a recurrence of pain at the operated site 10 days postoperatively. The pain was a dull ache, non-radiating, and showed a progressive increase in intensity. This was accompanied by constitutional symptoms like poor appetite and a general feeling of being unwell. In view of the non-specific nature of the complaints, no further investigation was carried out and the patient was advised to attend regular follow-ups. In our hospital, enhanced CT and X-rays were performed after thorough clinical evaluation. The CT scan revealed the malposition of a pedicle screw on both sides at the T11, T12, and L1 levels, which was a disaster. The left side T12 and L1 malpositioned screws had exited the lateral pedicle cortex and were clearly abutting the posteromedial aspect of the

*[source truncated at 250 words — full length: 402 words]*

</details>

<details>
<summary><strong>Gold reference summary</strong></summary>

In this paper, we report here a case in which inadvertent injury to the thoracic aorta resulted in pseudoaneurysm, its manifestation was initially vague, resulting in a delayed diagnosis. Delayed aortic pseudoaneurysm or injury can be asymptomatic for a long time. Patients with renewed or continued back pain should alert orthopaedic surgeons regarding the possibility of pseudoaneurysms, regardless of the period that has elapsed after pedicle screw implantation.

</details>

**D2 — BioBART LoRA** *(FK 10.2)*

A 21-year-old man presented to the emergency room (ER) complaining of stomach and serious back pain 2 months after spinal surgical intervention. Two months before his visit, he fell from a 10 m tall building. He was confirmed as suffering fracture and dislocation from T11 to L2 by a computed tomography (CT) examination. The patient complained of severe back pain. There was no neural deficit. He underwent thoracolumbar pedicle screw implant surgery using ‘freehand technique’ under fluoroscopic guidance in a local hospital. Eight pedicle screws with contoured rods were placed spanning T11 and L2 after the correction of deformities. The postoperative course was uneventful.

**Qwen-only** *(FK 10.0)*

A 21-year-old man experienced back and stomach pain after a fall from a building, leading to spinal surgery. The CT scan showed that two months post-surgery, some screws used in the thoracolumbar pedicle screw implant had moved out of the spine. They were pressing against the aorta near where an artery was forming an abnormally large bulge (pseudoaneurysm). This caused significant pain and other health issues like loss of appetite.

Further imaging revealed the screws had broken through the bone and touched or nearly touched the bulging artery inside the body. As this situation could be dangerous, another surgical team re-operated. During the operation, they found these problematic screws and removed them before repairing damage to the aorta with a new tube.

The patient recovered well from the major procedure but needs regular follow-ups.

**D3 — BioBART→MedJEx→Qwen rewrite** *(FK 6.2)*

A 21-year-old man came to the emergency room complaining of stomach pain and serious back pain 2 months after spinal surgical intervention. Two months ago, he fell from a 10-meter building. He was found to have a fracture and dislocation between vertebra T11 and L2 on a CT scan. The patient reported severe back pain. There were no problems with nerve function. He had surgery in a nearby hospital to put screws in his spine for these fractures. Doctors did this using a ‘freehand technique’ with help from x-rays showing where everything should go. They placed 8 screws shaped like rods to hold the vertebrae in place. Everything went well post-surgery.

### Analysis

| | D2 | Qwen | D3 |
|---|---|---|---|
| **Strengths** | | | |
| **Weaknesses** | | | |
| **Factual accuracy** | | | |
| **Readability** | | | |
| **Completeness** | | | |

**What a better system would do here:**

<!-- fill in -->

---

## Example 11 &nbsp;·&nbsp; ID 2085 &nbsp;·&nbsp; Group C

> Short source — systems diverge sharply on style

| Metric | D2 | Qwen | D3 | Gold |
|--------|:--:|:----:|:--:|:----:|
| FK Grade | 17.9 | 7.3 | 16.9 | 15.8 |
| Word count | 118 | 82 | 114 | 75 |
| Source words | 120 | | | |

**MLM jargon spans detected (13 total, top 12 shown):**
`a progressively enlarging mass`, `her right distal leg anterior aspect`, `a palpable, non-tender, subcutaneous mass`, `Ultrasonography`, `a well-defined hypoechoic lesion`, `surgical excision`, `Histopathological analysis`, `a poorly-circumscribed encapsulated neoplasm`, `myoepithelial cells`, `perivascular attenuation`, `slit`, `angioleiomyoma`

<details>
<summary><strong>Source (clinical case report)</strong></summary>

A 54-year-old female presented with a progressively enlarging mass in her right distal leg anterior aspect for 2 years, associated with occasional pain over the past 6 months. Physical examination revealed a palpable, non-tender, subcutaneous mass measuring approximately 2 cm in diameter, located in the medial aspect of the distal leg. Ultrasonography demonstrated a well-defined hypoechoic lesion with vascularity. Considering the suspicious nature of the mass, surgical excision was performed under regional anesthesia . Histopathological analysis revealed a well-circumscribed encapsulated neoplasm composed of myoepithelial cells proliferating with perivascular attenuation around slit such as vascular spaces, consistent with a diagnosis of angioleiomyoma . The patient recovered completely and was followed up at 6 monthly intervals for 2 years without any recurrence.

</details>

<details>
<summary><strong>Gold reference summary</strong></summary>

We present a case of angioleiomyoma in a 54-year-old female who presented with a palpable mass in her distal leg. The tumor was surgically excised, and histopathological examination confirmed the diagnosis of angioleiomyoma. In this article, we discuss the clinical presentation, diagnostic evaluation, and management of angioleiomyoma, with a focus on distal leg tumors. Furthermore, we provide a comprehensive review of the existing literature on angioleiomyomas, emphasizing findings and treatment outcomes reported in previous studies.

</details>

**D2 — BioBART LoRA** *(FK 17.9)*

A 54-year-old female presented with a progressively enlarging mass in her right distal leg anterior aspect for 2 years, associated with occasional pain over the past 6 months. Physical examination revealed a palpable, non-tender, subcutaneous mass measuring approximately 2 cm in diameter, located in the medial aspect of the distal lower leg. Ultrasonography demonstrated a well-defined hypoechoic lesion with vascularity. Considering the suspicious nature of tumor, surgical excision was performed under regional anesthesia. Histopathological analysis revealed a poorly-circumscribed encapsulated neoplasm composed of myoepithelial cells proliferating with perivascular attenuation around slit such as vascular spaces, consistent with a diagnosis of angioleiomyoma. The patient recovered completely and was followed up at 6 monthly intervals for 2 year without any recurrence.

**Qwen-only** *(FK 7.3)*

A 54-year-old lady had a lump in her leg for 2 years. The lump was painful and noticeable, growing larger. After an ultrasound exam showed a dark spot with blood vessels nearby, doctors decided to remove the lump under anesthesia. When they took out the lump, it was found to be made up of cells that build muscles around blood vessels. This condition is called angioleiomyoma. The woman got better after the surgery without any more lumps appearing over two years later.

**D3 — BioBART→MedJEx→Qwen rewrite** *(FK 16.9)*

A 54-year-old female presented with a progressively enlarging mass in her right distal leg anterior aspect for 2 years, associated with occasional pain over the past 6 months. Physical examination revealed a palpable, non-tender, subcutaneous mass measuring approximately 2 cm in diameter, located in the medial aspect of the distal lower leg. Ultrasonography showed a well-defined hypoechoic lesion with vascularity. Given the suspicious nature of the growth, surgical excision was done under regional anesthesia. The histopathological analysis indicated a poorly-circumscribed encapsulated neoplasm made up of myoepithelial cells showing perivascular attenuation around slit-like vascular spaces, consistent with an angioleiomyoma diagnosis. The patient recovered fully and was followed without any recurrence for the next 2 years.

### Analysis

| | D2 | Qwen | D3 |
|---|---|---|---|
| **Strengths** | | | |
| **Weaknesses** | | | |
| **Factual accuracy** | | | |
| **Readability** | | | |
| **Completeness** | | | |

**What a better system would do here:**

<!-- fill in -->

---

## Example 12 &nbsp;·&nbsp; ID 337 &nbsp;·&nbsp; Group C

> Qwen most readable; D2 more complete

| Metric | D2 | Qwen | D3 | Gold |
|--------|:--:|:----:|:--:|:----:|
| FK Grade | 17.2 | 8.0 | 14.0 | 17.0 |
| Word count | 286 | 109 | 226 | 191 |
| Source words | 439 | | | |

**MLM jargon spans detected (24 total, top 12 shown):**
`A 20-year-old female underwent deformity correction`, `presumed adult idiopathic scoliosis`, `pedicle screws`, `T3-T12`, `The anesthetic and scoliosis surgery`, `the intraoperative neuromonitoring`, `saturation`, `USG Doppler`, `bilateral lower limbs`, `computed tomography pulmonary angiography`, `pulmonary embolism`, `CTPA`

<details>
<summary><strong>Source (clinical case report)</strong></summary>

A 20-year-old female underwent deformity correction and spinal fusion surgery for presumed adult idiopathic scoliosis with posterior spinal instrumentation comprising pedicle screws and hooks extending from T3-T12 . The pre-operative assessment including an ECG did not demonstrate any abnormalities and her past medical history was uneventful while she had a normal level of physical activities including sports. The anesthetic and scoliosis surgery was uncomplicated with the patient maintaining good O2 levels and blood pressure while the intraoperative neuromonitoring showed stable motor and sensory potentials throughout. Her initial post-operative course was without incident until post-operative day 2 when she had episodes of acute de-saturation which prompted USG Doppler for bilateral lower limbs and computed tomography pulmonary angiography (CTPA). While both the studies ruled out any evidence of pulmonary embolism, CTPA revealed right heart strain pattern and an enlarged right side of the heart. A subsequent 2D echocardiography revealed a large atrial septal defect (ASD) which had remained asymptomatic up to that point. In addition, the CTPA demonstrated a lateral breech at the left T10 pedicle screw level with the screw tip being in close proximity to the aorta. A multidisciplinary team meeting was held where it was decided to prioritize cardiac management for ASD and keep the malpositioned screw under observation with removal planned for a later date. An ASD repair procedure was performed and this was uneventful with the patient making a good recovery. After recovery from the cardiac procedure, a repeat computed tomography (CT) scan was performed to plan

*[source truncated at 250 words — full length: 439 words]*

</details>

<details>
<summary><strong>Gold reference summary</strong></summary>

A 20-year-old female underwent posterior scoliosis correction and spinal fusion for an adult thoracic scoliosis with spinal instrumentation extending from T3-T12. In the immediate post-operative period, an episode of acute desaturation prompted a computed tomography pulmonary angiography (CTPA) to rule out pulmonary embolism. While CTPA demonstrated an enlarged right side of the heart (corresponding to a large atrial septal defect as evident on 2D-ECHO), an incidental lateral breech at the left T10 pedicle screw level was also identified with the screw tip in close proximity to the aorta. At this time, cardiac management was prioritized and a successful atrial septal defect repair procedure was performed. After recovery from the cardiac procedure, a plan was formulated with a multidisciplinary team to remove the pedicle screw. To determine the exact position of the screw tip in relation to the aorta, an angiogram and an intravascular ultrasound were done which showed no evidence of intra-luminal extension of the screw tip. In addition, a transesophageal ultrasound was performed and this ruled out any extra-luminal contact with the aorta. Subsequently, the spinal instrumentation was removed safely with the vascular surgeon and interventional radiologist being on standby.

</details>

**D2 — BioBART LoRA** *(FK 17.2)*

A 20-year-old female underwent deformity correction and spinal fusion surgery for presumed adult idiopathic scoliosis with posterior spinal instrumentation comprising pedicle screws and hooks extending from T3-T12. The pre-operative assessment including an ECG did not demonstrate any abnormalities and her past medical history was uneventful while she had a normal level of physical activities including sports. The anesthetic and scoliosis surgery was uncomplicated with the patient maintaining good O2 levels and blood pressure while the intraoperative neuromonitoring showed stable motor and sensory potentials throughout. Her initial post-operative course was without incident until post-operatively day 2 when she had episodes of acute de-saturation which prompted USG Doppler for bilateral lower limbs and computed tomography pulmonary angiography (CTPA). While both the studies ruled out any evidence of pulmonary embolism, CTPA revealed right heart strain pattern and an enlarged right side of the heart. A subsequent 2D echocardiography revealed a large atrial septal defect (ASD) which had remained asymptomatic up to that point. In addition, the CTPA demonstrated a lateral breech at the left T10 pedicle screw level and the screw tip was in close proximity to aorta. A multidisciplinary team meeting was held where it was decided to prioritize cardiac management for ASD and keep the malpositioned screw under observation with removal planned for a later date. The patient was followed in our clinic for an additional period of two years during which there was no clinical or radiographic evidence of scoliosis recurrence. A solid posterior spinal fusion was noted across the previously operated levels which allowed the spinal instrumentation to be removed without compromising the surgical outcome of scoliotic correction. Intraoperative neuro-monitoring was used and this procedure was uncomplicated. The fusion mass was re-grafted using allograft bone.

**Qwen-only** *(FK 8.0)*

A 20-year-old female had surgery to correct her scoliosis with screws and hooks. Her heart was checked before, during, and after the procedure, but nothing unusual was found. Post-surgery, she had trouble breathing properly, which led to tests that showed a hole in her heart (called an ASD) and the screws were close to her big artery (aorta). A repair for the ASD was done, and everything went well. Afterward, more tests showed the left screw might still be near the aorta. The screws were removed safely without further issues. Two years later, a second operation corrected any possible scoliosis recurrence. Overall, her surgeries went smoothly without major problems.

**D3 — BioBART→MedJEx→Qwen rewrite** *(FK 14.0)*

A 20-year-old female underwent deformity correction and spinal fusion surgery for presumed adult idiopathic scoliosis with posterior spinal instrumentation including pedicle screws and hooks from T3 to T12. Her pre-operative assessment, which included an ECG, didn’t show any abnormalities, and her medical history was good—she engaged in normal physical activities like sports. The anesthesia and spinal surgery went smoothly; she had stable oxygen levels and blood pressure during the operation. Intraoperatively, neuro-monitoring helped ensure no damage to nerve functions.

However, on post-op day 2, she experienced episodes of acute desaturation, prompting USG Doppler tests for her lower limbs and CTPA (computed tomography pulmonary angiography). While these exams ruled out pulmonary embolism, they indicated right heart strain and an enlarged right side of her heart. A subsequent echocardiogram showed a large atrial septal defect (ASD), previously undetected, but not causing symptoms up until then. Additionally, another CTPA revealed a breech at the left T10 screw level and the screw tip close to the aorta. 

A multidisciplinary meeting decided that managing the ASD should be prioritized. She returned to our clinic after two more years where there was no sign of scoliosis recurrence. A solid bone graft fused across the areas operated upon, allowing removal of hardware without compromising the spine’s correction. Intraoperative neuro-monitoring worked well without complications. The mass used for fusion was re-grated using donated bone.

### Analysis

| | D2 | Qwen | D3 |
|---|---|---|---|
| **Strengths** | | | |
| **Weaknesses** | | | |
| **Factual accuracy** | | | |
| **Readability** | | | |
| **Completeness** | | | |

**What a better system would do here:**

<!-- fill in -->

---

## Example 13 &nbsp;·&nbsp; ID 2991 &nbsp;·&nbsp; Group D

> D3 degraded vs D2 — pipeline failure mode

| Metric | D2 | Qwen | D3 | Gold |
|--------|:--:|:----:|:--:|:----:|
| FK Grade | 11.9 | 12.2 | 13.6 | 10.6 |
| Word count | 291 | 212 | 201 | 47 |
| Source words | 562 | | | |

**MLM jargon spans detected (18 total, top 12 shown):**
`human menopausal gonadotropin`, `a gynecologist`, `insomnia`, `tinnitus`, `tranquilizers`, `benzodiazepines`, `antidepressants`, `serotonin selective re-uptake inhibitors`, `SSRIs`, `her insomia`, `the supposed autonomic dysfunction`, `1 month`

<details>
<summary><strong>Source (clinical case report)</strong></summary>

A 37-year-old female complained of dizziness. She had started experiencing dizziness 3 years ago, following the infertility treatment that she had received. She experienced dizziness following an injection of human menopausal gonadotropin administered by a gynecologist and a visit psychologist. In addition to her dizziness, she also suffered from insomnia, tinnitus, and anxiety. Therefore, she was referred to a psychologist. However, the treatment of tranquilizers such as benzodiazepines and antidepressants such as serotonin selective re-uptake inhibitors (SSRIs) failed to cure the dizziness and only slightly improved her insomnia. She was therefore referred to our department for further examination and treatment. She expressed her dizziness as an event wherein she experienced frequent paroxysmal earthquakes occurring within seconds. The frequency of such episodes had recently increased to once every 5 minutes. She felt stable while doing her household tasks and she had never fallen. Audio-vestibular examination, including pure tone audiometry, posturography, and head MRI, revealed no abnormal findings. Her blood examination findings were normal; there was no spontaneous or evoked nystagmus. However, the peripheral part of her hand and foot often became pale due to poor peripheral circulation, an observation similar to Raynaud's phenomenon. She also experienced chronic headache and insomnia, whereby she woke up every 2 hours during the night. The results of the psychological examination were as follows: Self-rating Depression Scale (SDS), 47; Japanese version of the Cornell Medical Index (CMI), III; Manifest Anxiety Scale (MAS), 27; and Maudsley Obsessional-Compulsive Inventory (MOCI ), 9. MAS indicated a high level of

*[source truncated at 250 words — full length: 562 words]*

</details>

<details>
<summary><strong>Gold reference summary</strong></summary>

We present a case of a patient who suffered from phobic postural vertigo. A 37-year-old female complained of dizziness. She had started experiencing dizziness almost 3 years She was intractable to many sort of conventional therapy. In the end, her symptom disappeared after introduction of autogenic training.

</details>

**D2 — BioBART LoRA** *(FK 11.9)*

A 37-year-old female complained of dizziness. She had started experiencing dizziness 3 years ago, following the infertility treatment that she had received. She experienced dizziness following an injection of human menopausal gonadotropin administered by a gynecologist and a visit psychologist. In addition to her dizziness, she also suffered from insomnia, tinnitus, and anxiety. Therefore, she was referred to a psychologist. However, the treatment of tranquilizers such as benzodiazepines and antidepressants such as serotonin selective re-uptake inhibitors (SSRIs) failed to cure the dizziness and only slightly improved her insomia. We decided to focus on treating the patient's anxiety and the supposed autonomic dysfunction. After 1 month, AT was introduced by a clinical psychologist so as to ease her mental stress. The psychotherapy consisted of one 45-minute session every 3 weeks. The first session began with a brief introduction to the general background information about the cognitive approach, after which the patients was instructed how to perform AT. Thereafter, they performed AT in a relaxed sitting position on a chair for 10 minutes 3 times a day. No self-monitoring was advised. The patient was instructed to carry out slow and deep abdominal breathing at the beginning of AT and regular breathing during AT. She diligently and regularly continued this AT routine 3 time a day at her home. We prescribed setiptiline maleate and an additional herbal medicine, which is known to improve peripheral circulation. Astonishingly, after the introduction of AT, her mood stabilized and her dizziness decreased to less than one third. The dose of clotiazepam was reduced to 5 mg once a day for 2 weeks. However she didn't want to keep taking these drugs, since she wants to have a baby. No abnormality was reported in any physical examination, including posturography.

**Qwen-only** *(FK 12.2)*

A 37-year-old woman reported dizziness that had persisted for 3 years. She experienced it as frequent, brief "earthquakes" occurring within seconds. Along with the dizziness, she also had insomnia, tinnitus, and anxiety. Traditional medications such as tranquilizers and antidepressants didn't help her dizziness but slightly improved her sleep issues.

Tests like hearing tests, balance tests and MRI did not show any abnormalities. No other health conditions seemed to be causing her dizziness or symptoms. The psychologist found high levels of anxiety, depression, and was diagnosed with psychosomatic phobic postural vertigo. 

We prescribed setiptiline to improve circulation, along with herbal remedies. After a couple weeks, though, she discontinued these due to wanting to conceive again. Physical exams revealed no abnormal physical causes yet her blood test findings were normal; however she showed signs of poor peripheral circulation similar to Raynaud’s phenomenon.

Psychotherapy including teaching 'Attention Training' (AT) helped significantly. This involved performing slow deep breathing exercises while sitting on a chair for ten minutes three times daily. AT consisted of focusing on present experiences without judgment. With continued adherence to this exercise routine, her mood improved and all symptoms disappeared in a short time.

Follow-up visits six months later confirmed no recurring symptoms. Anxiety was reduced, peripheral circulation improved and overall well-being returned.

**D3 — BioBART→MedJEx→Qwen rewrite** *(FK 13.6)*

A 37-year-old female complained of dizziness. She had started experiencing dizziness 3 years ago, following infertility treatment. The dizziness occurred after an injection of human menopausal gonadotropin given by a gynecologist and a visit to a psychologist for insomnia and tinnitus. Despite taking tranquilizers like benzodiazepines and antidepressants like SSRIs, the dizziness did not improve significantly; her insomnia improved slightly. Therefore, she was referred to a psychologist. Therapy focused on treating anxiety and supposed autonomic dysfunction through a technique called setiptiline maleate along with an herbal medicine known to improve peripheral circulation. Treatment also included anxiety management exercises called "Attention Training" (AT) conducted by a clinical psychologist. AT sessions lasted 45 minutes every 3 weeks and consisted of 10-minute practices done 3 times daily in a relaxed sitting position. Each session involved slow, deep abdominal breathing at the start and regular breathing towards the end. Physical examinations, including posturography, showed no abnormalities.

Setiptiline maleate and the herbal medicine were prescribed alongside reduced doses of clotiazepam to 5 mg per day for 2 weeks. However, she stopped taking these medications due to her desire to have a baby. Her mood stabilized, and her dizziness decreased significantly—about one third less compared to before.

### Analysis

| | D2 | Qwen | D3 |
|---|---|---|---|
| **Strengths** | | | |
| **Weaknesses** | | | |
| **Factual accuracy** | | | |
| **Readability** | | | |
| **Completeness** | | | |

**What a better system would do here:**

<!-- fill in -->

---

## Example 14 &nbsp;·&nbsp; ID 2546 &nbsp;·&nbsp; Group E

> Long source (1600 words) — compression under pressure

| Metric | D2 | Qwen | D3 | Gold |
|--------|:--:|:----:|:--:|:----:|
| FK Grade | 13.3 | 15.6 | 13.7 | 11.9 |
| Word count | 197 | 160 | 185 | 107 |
| Source words | 1605 | | | |

**MLM jargon spans detected (24 total, top 12 shown):**
`a 51-year-old white man`, `severe coronary artery disease`, `cardiogenic shock`, `one point`, `reduced ejection fraction`, `20%–25%`, `hypertension`, `his brother`, `ataxia`, `hyponatremia`, `hyponatremic dehydration`, `an elevated brain natriuretic peptide`

<details>
<summary><strong>Source (clinical case report)</strong></summary>

The patient is a 51-year-old white man with a medical history significant for severe coronary artery disease resulting in cardiogenic shock (at one point requiring a short-term left ventricular assist device), heart failure with reduced ejection fraction estimated at 20%–25%, hypertension and schizophrenia who presented with his brother to the emergency department with 5-day history of fatigue, ataxia, altered mental status and abnormal behaviour. His brother states that he has been getting steadily worse, with multiple near falls and unsteadiness in 2 days leading up to admission. He has never had any symptoms like this before, and has poor insight into his symptoms. His behaviour and mental status have also been inappropriate, as the brother states he has been unable to complete activities of daily living such as bathing or feeding himself. His affect has also been more flat than normal, with the patient’s brother reporting significantly longer time to respond to questions than his baseline. He also reported one episode of hyper elation and agitation that resolved after several hours. He has been alert and oriented to only person and place, but not time (normally he is fully oriented). The patient denies any shortness of breath, chest pain, nausea, headache, vomiting, diarrhoea or decreased oral intake. He and his brother deny any trauma. The patient denies any hallucinations, delusions, depressed mood or worsening negative symptoms of schizophrenia. His medication regimen is notable for spironolactone 25 mg daily, torsemide 200 mg two times per day and metolazone 2.5 mg daily

*[source truncated at 250 words — full length: 1605 words]*

</details>

<details>
<summary><strong>Gold reference summary</strong></summary>

A 51-year-old man was hospitalised for severe hyponatremia. Initial history and physical examination suggested hypovolemia, and he was treated with normal saline at 100 mL/hour. After several days, his hyponatremia failed to improve, and then worsened without resolution of presenting ataxia and fatigue. He had no new complaints including no cough or orthopnea. He had no jugular venous distention or oedema, and his lungs were clear to auscultation. Point-of-care ultrasound was used, revealing a distended inferior vena cava, pulmonary oedema and pleural effusion, suggesting hypervolemia. Based on ultrasound findings, we treated with 60 mg oral torsemide two times per day. Hyponatremia resolved without complication within 48 hours.

</details>

**D2 — BioBART LoRA** *(FK 13.3)*

We present a 51-year-old white man with a medical history significant for severe coronary artery disease resulting in cardiogenic shock (at one point requiring a short-term left ventricular assist device), heart failure with reduced ejection fraction estimated at 20%–25%, hypertension and schizophrenia who presented with his brother to the emergency department with 5-day history of fatigue, ataxia, altered mental status and abnormal behaviour. He was found to have hyponatremia, hyponatremic dehydration and an elevated brain natriuretic peptide (BNP). The patient was treated with spironolactone, torsemide 200 mg two times per day and metolazone 2.5 mg daily for diuresis. His most recent lipid panel from 8 months prior had a total cholesterol of 263 mg/dL, high-density lipoprotein (HDL) of 37 mg/dL and a creatinine of 2.32 mg /dL. The patient and his brother denied any trauma. On physical examination, he was alert and oriented, with no focal neurologic deficit, and a flat affect. He had very dry mucous membranes, decreased skin turgor and no oedema. His cardiac examination was notable for an irregular rhythm, no appreciated jugular venous distention with equal peripheral pulses. His pulmonary examination was overall clear to auscultation. Abdomen was non-distended, with normal bowel sounds.

**Qwen-only** *(FK 15.6)*

A 51-year-old man with severe heart disease, including cardiogenic shock from coronary artery disease and congestive heart failure, presented to the hospital with symptoms of fatigue, ataxia, altered behavior, and unsteadiness. His hyponatremia (serum sodium of 108 mmol/L) was severe and suspected due to overuse of diuretics for his condition. He eventually required a slow sodium correction regimen with isotonic fluids. Despite improving after initial management and holding his olanzapine to prevent worsening hyponatremia, he showed some residual symptoms on discharge.

Upon further assessment using point-of-care ultrasound, it was discovered the patient had hypervolemia instead of euvolemia. This led to an addition of a diuretic regimen and more careful monitoring. His sodium improved significantly, and he was safely discharged home with close follow-up recommended.

During subsequent care, there were challenges managing medication adherence due to his mental health history and difficulty understanding his complex medical issues. While stable initially, he experienced ongoing symptoms without return visits or regular doctor’s appointments.

**D3 — BioBART→MedJEx→Qwen rewrite** *(FK 13.7)*

We present a 51-year-old white man with a medical history significant for severe coronary artery disease resulting in cardiogenic shock (at one point requiring a short-term left ventricular assist device), heart failure with reduced ejection fraction estimated at 20%–25%, hypertension and schizophrenia who presented with his brother to the emergency department with 5-day history of fatigue, ataxia, altered mental status, and abnormal behaviour. He was found to have hyponatremia, hyponatremic dehydration, and an elevated brain natriuretic peptide (BNP). The patient was treated with spironolactone, torsemide 200 mg twice daily, and metolazone 2.5 mg daily for diuresis. His most recent lipid panel from 8 months prior had a total cholesterol of 263 mg/dL, high-density lipoprotein (HDL) of 37 mg/dL, and a creatinine of 2.32 mg/dL. The patient and his brother denied any trauma. On physical examination, he was alert and oriented and had a flat affect. His skin was very dry with decreased turgor; he did not have oedema. Abdominal auscultation revealed normal bowel sounds, but on cardiac examination, there was an irregular rhythm with no jugular venous distension or unequal pulses. Pulmonary examination was clear.

### Analysis

| | D2 | Qwen | D3 |
|---|---|---|---|
| **Strengths** | | | |
| **Weaknesses** | | | |
| **Factual accuracy** | | | |
| **Readability** | | | |
| **Completeness** | | | |

**What a better system would do here:**

<!-- fill in -->

---

## Example 15 &nbsp;·&nbsp; ID 1866 &nbsp;·&nbsp; Group E

> Long source — D3 achieves best FK here

| Metric | D2 | Qwen | D3 | Gold |
|--------|:--:|:----:|:--:|:----:|
| FK Grade | 18.7 | 14.9 | 11.5 | 15.2 |
| Word count | 154 | 164 | 164 | 151 |
| Source words | 1579 | | | |

**MLM jargon spans detected (19 total, top 12 shown):**
`An 83-year-old Caucasian male`, `erysipelas`, `both legs`, `COVID-19 symptoms`, `diarrhea`, `myocardial infarction`, `percutaneous coronary intervention`, `stent implantation`, `permanent atrial fibrillation`, `stable angina pectoris`, `non-Hodgkin's lymphoma`, `symptomatic treatment`

<details>
<summary><strong>Source (clinical case report)</strong></summary>

An 83-year-old Caucasian male was hospitalized at a local hospital for erysipelas on both legs. Upon admission, the patient was routinely tested for COVID-19 by a reverse transcriptase-polymerase chain reaction test and was found to be positive. The patient was not vaccinated against COVID-19. While in the hospital, the patient did not develop any COVID-19-related symptoms. After receiving treatment for erysipelas for seven days, the patient was stable and was discharged. While at home, COVID-19 symptoms—fever, cough, weakness, and diarrhea appeared, gradually worsened, and 13 days after the onset of symptoms, the patient was admitted to the Latvian Centre of Infectious Diseases (LCID). Upon admission to LCID, the patient did not display any neurological problems as contact with the patient could be established, and he could orient in time and space. The skin and mucous membranes were pale and dry, with signs of recurrent erysipelas on both legs. During auscultation, diffuse crackles on both sides of the lungs were found, and palpation of the abdomen did not show any pathology, including no signs of dysuria; the patient had mild peripheral edema on both legs. The patient had a febrile body temperature of 38.0 ºC, tachycardia with a heart rate of 100/min, blood pressure was 110/70 mmHg, and respiratory rate was 24/min with blood oxygen saturation of 89% on room air. The patient was diagnosed with clinically severe COVID-19 with bilateral pneumonia, as determined with radiological imaging , and respiratory failure type I . From the second to the fourth day

*[source truncated at 250 words — full length: 1579 words]*

</details>

<details>
<summary><strong>Gold reference summary</strong></summary>

Here we describe a fatal COVID-19 case of an 83-year-old Caucasian male patient with various underlying comorbidities, including cardiovascular and autoimmune disorders, as well as immunosuppression due to lymphoma treatment. Upon admission, the patient was radiologically diagnosed with severe COVID-19. The patient was febrile and presented with diarrhea, continued dyspnea, tachypnea, and low blood oxygen saturation, treated with high-concentration oxygen supplementation and antibacterial therapy. Overall the patient was treated for COVID-19 for 19 days. Blood tests were performed upon admission, on the fifth, 10th, 13th, and 19th day. In addition, nasopharyngeal swab, blood, urine, and fecal samples were collected from the patient on the 14th day for virological and immunological investigations. Severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) was detectable in all samples collected from this patient, including blood plasma and peripheral blood mononuclear cells (PBMC), with very high viral loads. However, neither virus-specific IgA, IgM, nor IgG antibodies were detectable.

</details>

**D2 — BioBART LoRA** *(FK 18.7)*

An 83-year-old Caucasian male was hospitalized at a local hospital for erysipelas on both legs. While at home, COVID-19 symptoms—fever, cough, weakness, and diarrhea appeared, gradually worsened, and 13 days after the onset of symptoms, the patient was admitted to the Latvian Centre of Infectious Diseases (LCID). The patient had a history of cardiovascular problems (coronary heart disease, myocardial infarction, followed by percutaneous coronary intervention with stent implantation, permanent atrial fibrillation, stable angina pectoris and a myriad of other chronic health problems, including non-Hodgkin's lymphoma) and was treated with antibiotics and symptomatic treatment. On the tenth day of hospitalization, he presented with recurrent febrile body temperature of 38.5–39.0 ºC, severe malaise, disorientation, and blood oxygen desaturation. In addition, laboratory findings showed significantly elevated inflammatory markers, and the decision was made to switch to antibacterial therapy and to start oxygen supplementation through a high-concentration oxygen mask with a reservoir bag of 15 L per minute.

**Qwen-only** *(FK 14.9)*

An 83-year-old male with a history of heart disease and several other chronic illnesses was hospitalized for erysipelas, testing positive for COVID-19. After initial treatment, he was discharged but had worsened symptoms and was admitted to an infectious diseases hospital due to clinically severe COVID-19 with bilateral pneumonia. Despite antibiotic treatment, C. difficile infection was diagnosed, causing further diarrhea complications. The patient's condition deteriorated, requiring high-flow oxygen support and steroids.

The patient died from advanced COVID-19-related lung damage after being treated for 19 days. The medical care included dual antibacterial therapy, dexamethasone, and various supportive treatments like fluids and pain management. Blood tests showed high inflammation markers and SARS-CoV-2 detection in all samples (nasopharyngeal swab at very high load; feces also highly loaded). Elevated inflammatory cytokines, including IL-6 and IL-18, were confirmed, despite serology showing negative antibody levels indicating the absence of prior infection.

The patient received multiple forms of oxygen support and corticosteroids, highlighting his severe case of respiratory distress secondary to COVID-19.

**D3 — BioBART→MedJEx→Qwen rewrite** *(FK 11.5)*

An 83-year-old Caucasian male was hospitalized at a local hospital for erysipelas on both legs. While at home, he got COVID-19 symptoms such as fever, cough, weakness, and diarrhea. These symptoms became worse over 13 days and he was admitted to the Latvian Centre of Infectious Diseases (LCID). The patient had heart problems (coronary heart disease, heart attack, followed by doing a procedure called percutaneous coronary intervention and getting a stent put in, having permanent atrial fibrillation, angina that wasn't life-threatening, other health issues like lymphoma) and was treated with antibiotics and other treatments to feel better. After ten days in the hospital, his temperature kept going up to 38.5–39.0 degrees Celsius, he felt very weak, confused, and his blood oxygen levels were low. Lab tests showed high levels of things that show there’s an infection. So doctors switched to stronger antibiotics and started giving him more oxygen through a machine that keeps track of how much oxygen is in the air you breathe.

### Analysis

| | D2 | Qwen | D3 |
|---|---|---|---|
| **Strengths** | | | |
| **Weaknesses** | | | |
| **Factual accuracy** | | | |
| **Readability** | | | |
| **Completeness** | | | |

**What a better system would do here:**

<!-- fill in -->

---

## Example 16 &nbsp;·&nbsp; ID 3265 &nbsp;·&nbsp; Group E

> Long source, highest jargon count (32 spans)

| Metric | D2 | Qwen | D3 | Gold |
|--------|:--:|:----:|:--:|:----:|
| FK Grade | 13.2 | 15.6 | 13.4 | 14.9 |
| Word count | 264 | 159 | 241 | 199 |
| Source words | 1538 | | | |

**MLM jargon spans detected (32 total, top 12 shown):**
`A 20-year-old male patient`, `diarrhea`, `peritoneal stimulation sign`, `the diaphragm`, `diagnostic abdominal puncture`, `5 mL`, `perforation`, `Laparotomy`, `exploration`, `diffuse peritonitis, acute gangrenous appendicitis`, `perforation`, `adhesive intestinal obstruction`

<details>
<summary><strong>Source (clinical case report)</strong></summary>

A 20-year-old male patient was admitted to our hospital on July 9, 2021, for evaluation of recurring abdominal pain, diarrhea, and black stools, which had been persisting for 6 years. The patient received treatment at a local hospital 6 years back for seven days of unexplained abdominal pain. On examination, the peritoneal stimulation sign was positive. Abdominal X-ray at the time revealed a gas shadow located below the diaphragm. On diagnostic abdominal puncture, 5 mL of purulent fluid was withdrawn, thereby raising the suspicion of perforation in the digestive tract. Laparotomy performed for exploration revealed diffuse peritonitis, acute gangrenous appendicitis with perforation, adhesive intestinal obstruction, and pelvic abscess. Appendectomy, intestinal adhesiolysis, and pelvic abscess removal surgery were performed during the procedure (July 2, 2015). Following surgery, the patient’s condition improved and he was discharged. However, one year after discharge, the patient stared experiencing recurrent abdominal pain and dark red loose stools, with positive test results for fecal occult blood and mild to moderate anemia. Colonoscopy had been performed several times before and showed the presence of intestinal ulcers and bleeding at other hospitals. In light of the colonoscopy findings, a clinical diagnosis of IBD was made. Oral treatment with mesalazine was administered at a local hospital, at a dose of 2 tablets four times a day for a total of 6 weeks. However, there was no improvement in the symptoms of rectal bleeding. Subsequently, oral treatment with azathioprine was added, at a dose of 50 mg once a day for

*[source truncated at 250 words — full length: 1538 words]*

</details>

<details>
<summary><strong>Gold reference summary</strong></summary>

A 20-year-old man was admitted to our center with a 6-year history of recurrent abdominal pain, diarrhea, and dark stools. At the onset 6 years ago, the patient had received treatment at a local hospital for abdominal pain persisting for 7 d, under the diagnosis of diffuse peritonitis, acute gangrenous appendicitis with perforation, adhesive intestinal obstruction, and pelvic abscess. The surgical treatment included exploratory laparotomy, appendectomy, intestinal adhesiolysis, and pelvic abscess removal. The patient's condition improved and he was discharged. However, the recurrent episodes of abdominal pain and passage of black stools started again one year after discharge. On the basis of these features and results of subsequent colonoscopy, the clinical diagnosis was established as inflammatory bowel disease (IBD). Accordingly, aminosalicylic acid, immunotherapy, and related symptomatic treatment were administered, but the symptoms of the patient did not improve significantly. Further investigations revealed mutations in the ACVRL1 and PLA2G4A genes. ACVRL1 and PLA2G4A are involved in angiogenesis and coagulation, respectively. This suggests that the chronic intestinal ulcers and bleeding in this case may be linked to mutations in the ACVRL1 and PLA2G4A genes. Oral Kangfuxin liquid was administered to promote healing of the intestinal mucosa and effectively manage clinical symptoms.

</details>

**D2 — BioBART LoRA** *(FK 13.2)*

A 20-year-old male patient was admitted to our hospital on July 9, 2021, for evaluation of recurring abdominal pain, diarrhea, and black stools, which had been persisting for 6 years. On examination, peritoneal stimulation sign was positive. Abdominal X-ray at the time revealed a gas shadow located below the diaphragm. On diagnostic abdominal puncture, 5 mL of purulent fluid was withdrawn, thereby raising the suspicion of perforation in the digestive tract. Laparotomy performed for exploration revealed diffuse peritonitis, acute gangrenous appendicitis with perforation, adhesive intestinal obstruction, and pelvic abscess. Appendectomy, intestinal adhesiolysis, and abdominal abscess removal surgery were performed during the procedure (July 2, 2015). Following surgery, the patient’s condition improved and he was discharged. However, one year after discharge, he stared experiencing recurrent abdominal pain and dark red loose stools, with positive test results for fecal occult blood and mild to moderate anemia. Colonoscopy had been performed several times before and showed the presence of intestinal ulcers and bleeding at other hospitals. In light of the colonoscopy findings, a clinical diagnosis of IBD was made. Oral treatment with mesalazine was administered at a local hospital, at a dose of 2 tablets four times a day for a total of 6 weeks. Despite this treatment, symptoms of rectal bleeding did not improve. The Mayo score was 8 points both before and after medication (with an increase of 2-3 times per day compared to the normal bowel movements, mixed blood in stool within less than half of time, ulcer formation detected by endoscopy, and moderate condition). Considering its ineffectiveness, mesalazine and azathioprine treatment was discontinued.

**Qwen-only** *(FK 15.6)*

A 20-year-old male patient presented with recurring abdominal pain, diarrhea, and black stools for six years. He previously underwent surgery for acute gangrenous appendicitis with perforation a year prior. Despite treatment, including mesalazine and azathioprine, his symptoms persisted and worsened.

Upon hospitalization in July 2021, he had increased episodes of black stools about 3-4 times daily, along with anemia. Further testing revealed intestinal ulcers, mixed blood in the stool, ulcer formations detected by endoscopy, and mild-to-moderate anemia. Colonoscopy showed chronic non-atrophic gastritis due to bile reflux. Pathological examination from the ileum indicated superficial ulceration with inflammation and granulomas. The histopathology of the colon suggested moderate-to-severe chronic inflammation.

After initial treatments were ineffective, further genomic testing identified ACVRL1 and PLA2G4A gene mutations linked to genetic disorders affecting intestinal mucosal healing. Kangfuxin liquid was prescribed to enhance tissue repair, resulting in symptom alleviation upon initiation and continued use after discharge. This modified approach helped manage his condition effectively without recurrent exacerbations.

**D3 — BioBART→MedJEx→Qwen rewrite** *(FK 13.4)*

A 20-year-old male patient was admitted to our hospital on July 9, 2021, for evaluation of recurring abdominal pain, diarrhea, and dark red loose stools, which had been persistent for 6 years. On physical examination, peritoneal stimulation sign was positive. An abdominal X-ray showed a gas shadow below the diaphragm. During diagnostic abdominal puncture, 5 mL of pus-like fluid was withdrawn, indicating possible perforation of the digestive tract. Laparotomy for exploration revealed diffuse peritonitis, acute gangrenous appendicitis with perforation, adhesive intestinal obstruction, and pelvic abscess. Appendectomy, intestinal adhesiolysis, and removal of abdominal abscess were performed (July 2, 2015). After surgery, the patient’s condition improved and was discharged. However, one year later, he started experiencing recurrent abdominal pain and dark red loose stools, along with results that tested positive for fecal occult blood and mild to moderate anemia. Colonoscopy done previously indicated presence of intestinal ulcers and bleeding at other hospitals. Given colonoscopy findings, a clinical diagnosis of IBD was made. Mesalazine treatment was prescribed by a local hospital in doses of 2 tablets four times a day for 6 weeks, but it did not improve rectal bleeding symptoms. The Mayo score remained at 8 points before and after medication (with a significant increase of up to twice per day compared to normal bowel movements, mixed blood in stool less than half of the time, detected ulcers by endoscopy, and moderate condition). Due to its ineffectiveness, mesalazine and azathioprine treatments were stopped.

### Analysis

| | D2 | Qwen | D3 |
|---|---|---|---|
| **Strengths** | | | |
| **Weaknesses** | | | |
| **Factual accuracy** | | | |
| **Readability** | | | |
| **Completeness** | | | |

**What a better system would do here:**

<!-- fill in -->

---

## Aggregate Patterns

*Fill in after reading all 16 examples.*

### D2 (BioBART LoRA)

**Consistent strengths:**
- 

**Consistent weaknesses:**
- 

### Qwen-only

**Consistent strengths:**
- 

**Consistent weaknesses:**
- 

### D3 pipeline

**Consistent strengths:**
- 

**Consistent weaknesses:**
- 

---

## Direction for a Better System

*Write 3–5 concrete ideas here after completing the analysis above.*

1. 
2. 
3. 
