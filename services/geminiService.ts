// src/services/geminiService.tsx
import { GoogleGenAI, GenerateContentResponse, GroundingChunk } from "@google/genai";
import type { UploadedFile, MockupLevel, GroundingSource, Estimation, InvestmentAnalysis } from '../types';

const getAi = () => {
	const apiKey = import.meta.env.VITE_API_KEY as string | undefined;
	if (!apiKey) {
		throw new Error("Missing VITE_API_KEY. Add it in Vercel → Project → Settings → Environment Variables.");
	}
	return new GoogleGenAI({ apiKey });
};

const fileToGenerativePart = (file: UploadedFile) => {
	return {
		inlineData: {
			data: file.base64,
			mimeType: file.type,
		},
	};
};

export const getRehabEstimate = async (address: string, files: UploadedFile[], finishLevel: MockupLevel): Promise<{ markdown: string; sources: GroundingSource[] }> => {
	const ai = getAi();
	const model = 'gemini-2.5-flash';

	const prompt = `
        System Instruction: You are an expert real-estate rehab estimator. Your task is to provide a detailed, area-by-area rehabilitation cost estimate based on photos of a property at "${address}".

        **Crucial Step: Use Google Search to find local contractor pricing and material costs for the region around "${address}" to ensure accuracy.**

        All cost estimates should be tailored to a "${finishLevel}" finish level and be as precise as possible, aiming for a tight range of +/- 5%.

        User Prompt:
        Please analyze the provided photos and generate a detailed rehabilitation estimate. Follow this structure precisely and provide your output in markdown format.

        1.  **Project Summary:**
            *   Provide a total estimated cost range for the entire project. This range must be tight (+/- 5%) and based on your search for local pricing.
            *   Give an overall project difficulty rating on a 1-5 scale (1 = simple cosmetic, 5 = major structural/permit work).
            *   List any key assumptions you're making (e.g., "assuming no hidden water damage behind walls," "cost estimates are for mid-market labor in the region").
            *   **Key Risks:** Identify and list the 2-3 biggest risks based on the visual evidence. Example: "The 20% contingency is non-negotiable due to the high probability of finding extensive subfloor rot in Bathroom 2 and unpermitted wiring in the addition."
            *   **Actionable Advice:** Provide clear, imperative next steps for the investor. Example: "Strongly recommend getting firm bids from a licensed General Contractor, electrician, and plumber *before* closing. This property should not be purchased without these professional on-site assessments."

        2.  **Itemized Breakdown:**
            *   Create a markdown table with the following columns: "Area", "Observations", "Recommendations", "Estimated Cost", "Difficulty (1-5)".
            *   Walk through each key area of the property visible in the photos (e.g., Exterior, Roof, Kitchen, Bathroom 1, Living Room, Foundation, Electrical, Plumbing, etc.).
            *   For each area:
                *   **Observations:** Describe what you see, noting any visible damage, wear, or defects.
                *   **Recommendations:** Suggest specific repairs or replacements needed.
                *   **Estimated Cost:** Give a ballpark cost for the recommended work, grounded in local pricing for a "${finishLevel}" finish level.
                *   **Difficulty (1-5):** Rate the complexity of the work for that specific area.

        **Output Format (Strict Markdown):**

        ### Project Summary
        **Total Estimated Cost:** [e.g., $55,000 - $60,000]
        **Overall Difficulty:** [e.g., 4]
        **Assumptions:**
        *   [Assumption 1]
        *   [Assumption 2]
        **Key Risks:**
        *   [Risk 1]
        *   [Risk 2]
        **Actionable Advice:**
        *   [Advice 1]
        *   [Advice 2]

        ### Itemized Breakdown
        | Area | Observations | Recommendations | Estimated Cost | Difficulty (1-5) |
        | :--- | :--- | :--- | :--- | :--- |
        | [e.g., Kitchen] | [e.g., Dated oak cabinets, laminate countertops are peeling.] | [e.g., Replace all cabinets and countertops. Install new sink and faucet.] | [e.g., $12,500 - $13,800] | [e.g., 3] |
        | [Next Area] | ... | ... | ... | ... |
    `;

	const imageParts = files.map(fileToGenerativePart);

	const result: GenerateContentResponse = await ai.models.generateContent({
		model,
		contents: { parts: [{ text: prompt }, ...imageParts] },
		config: {
			temperature: 0.0,
			tools: [{ googleSearch: {} }],
		},
	});

	if (!result.text) {
		console.error("Rehab estimate generation failed. The model returned an empty text response. Full response:", JSON.stringify(result, null, 2));
		throw new Error("The AI model returned an empty response for the rehab estimate. This may be due to a content filter or an internal error.");
	}

	const groundingChunks = result.candidates?.[0]?.groundingMetadata?.groundingChunks ?? [];
	const sources: GroundingSource[] = groundingChunks
		.filter((chunk: GroundingChunk) => chunk.web?.uri && chunk.web?.title)
		.map((chunk: GroundingChunk) => ({
			uri: chunk.web.uri,
			title: chunk.web.title,
		}));

	return { markdown: result.text, sources };
};

export const getInvestmentAnalysis = async (
	address: string,
	estimation: Estimation,
	purchasePrice: string
): Promise<InvestmentAnalysis> => {
	const ai = getAi();
	const model = 'gemini-2.5-flash';
	const totalRepairCost = estimation.summary.totalEstimatedCost;

	const prompt = `
        **System Instruction:** You are an expert real estate investment analyst. Your task is to provide a comprehensive investment analysis for the property at "${address}", given the estimated rehabilitation costs.

        **CRUCIAL - Multi-Step Process for Finding Comps:**
        1.  **Identify Property Details & Neighborhood Boundaries:**
            *   First, use Google Search to find the **"Year Built"** for the subject property at **"${address}"**.
            *   Second, use Google Search/Maps to identify the specific **subdivision name** of the property.
            *   Third, identify any **major highways or roads** that act as clear boundaries for this subdivision. Comps should NOT cross these barriers.

        2.  **Find Comps with Quality-First Fallback Logic:** Your goal is to find 1-3 *highly relevant* comparable sales (comps) to determine an accurate After Repair Value (ARV). Quality is more important than quantity. Follow this search process:
            *   **Attempt 1 (Ideal Criteria):** Search for comps meeting ALL of these strict criteria:
                *   **Neighborhood:** Located within the **same subdivision** and **NOT separated by a major highway/road**. This is the most important rule.
                *   **Recency:** Sold within the last **6 months** from today's date.
                *   **Proximity:** Located within a **0.5-mile radius** of the subject property.
                *   **Age:** Built within **+/- 10 years** of the subject property's Year Built.
            *   **Attempt 2 (Relax Proximity):** If you cannot find at least 1-2 comps, relax the proximity to a **1-mile radius** and search again, but strictly maintain all other criteria (same subdivision, no major barriers, 6 months recency, +/- 10 years built).
            *   **Attempt 3 (Relax Recency):** If you still cannot find at least 1-2 comps, relax the recency to **sold within the last 12 months** and search again, keeping the proximity at 1 mile and the age/neighborhood criteria the same.

        3.  **Report Your Findings:** You MUST populate the \`compsSearchCriteria\` field in the JSON output with a clear statement explaining which criteria were used to find the comps (e.g., "Comps were found using the ideal criteria," or "Comps search criteria were relaxed to a 1-mile radius to find sufficient results."). This is not optional. If you find fewer than 3 comps, that is acceptable as long as they are high quality.

        **Property Information:**
        *   **Address:** ${address}
        *   **Estimated Rehab Cost:** ${totalRepairCost}
        *   **Property Condition Summary (from previous analysis):** ${estimation.repairs.map(r => `${r.area}: ${r.observations}`).join('. ')}

        **Your Task:**
        Generate a complete investment analysis. Provide your output as a single JSON object inside a markdown code block. Adhere strictly to the schema provided below.

        **GUIDANCE FOR ANALYSIS:**
        *   **investorFit.analysis:** Provide a neutral, data-driven analysis of the property as an investment. Discuss the relationship between the After Repair Value (ARV) and the rehab costs. Mention that investors often use formulas like the 70% rule to calculate a Maximum Allowable Offer (MAO). This rule is a baseline and can be adjusted for market conditions. Do NOT make a final judgment on whether the deal is "good" or "bad"; just present the facts. Set "fitsCriteria" to a placeholder value of \`true\`.
        *   **exitStrategies:** When discussing "Fix and Flip", frame it in the context of acquiring a property at a discount to its ARV to allow for profit after rehab costs. For "Buy and Hold" or "BRRRR" strategies, introduce and briefly explain the importance of follow-up analysis using key buy-and-hold metrics like the **1% Rule** for initial rent screening, **Cash-on-Cash Return** (which accounts for financing), and **DSCR (Debt Service Coverage Ratio)** for loan qualification.

        **Output Format (Strict JSON):**
        \`\`\`json
        {
          "suggestedARV": "...",
          "estimatedRepairCost": "${totalRepairCost}",
          "investorFit": {
            "fitsCriteria": true,
            "analysis": "..."
          },
          "propertyCondition": "...",
          "estimatedRepairLevel": "...",
          "compsSearchCriteria": "...",
          "comparables": [
            {
              "address": "...",
              "soldDate": "...",
              "soldPrice": "...",
              "sqft": "...",
              "bedBath": "..."
            }
          ],
          "exitStrategies": [
            {
              "strategy": "...",
              "details": "..."
            }
          ]
        }
        \`\`\`

        **Field Explanations:**
        *   **suggestedARV:** (String or Number) After Repair Value. A dollar amount based on your search for comparable sales.
        *   **investorFit:** (Object) - See "GUIDANCE FOR ANALYSIS" above.
        *   **propertyCondition:** (String) A 1-2 sentence summary of the property's overall condition based on the provided summary.
        *   **estimatedRepairLevel:** (String) Classify the rehab level. Must be one of: 'Light Cosmetic', 'Medium', 'Heavy', 'Gut'.
        *   **compsSearchCriteria:** (String) A sentence explaining the criteria used to find the comps (e.g., ideal, relaxed radius, relaxed recency).
        *   **comparables:** (Array of Objects) List 1-3 recent comparable sales you found via Google Search.
        *   **exitStrategies:** (Array of Objects) Propose 2-3 viable exit strategies with brief explanations, following the guidance above.
    `;

	const result = await ai.models.generateContent({
		model,
		contents: prompt,
		config: {
			temperature: 0.1,
			tools: [{ googleSearch: {} }],
		}
	});

	if (!result.text) {
		console.error("Investment analysis generation failed. The model returned an empty text response. Full response:", JSON.stringify(result, null, 2));
		throw new Error("The AI model returned an empty response for the investment analysis. This may be due to a content filter or an internal error.");
	}

	try {
		const jsonMatch = result.text.match(/```json\n([\s\S]*?)\n```/);
		if (jsonMatch && jsonMatch[1]) {
			const parsedJson = JSON.parse(jsonMatch[1]);

			// --- Start of Application-Side Business Logic ---

			const parseCurrency = (value: string | number): number => {
				if (typeof value === 'number') return value;
				if (typeof value !== 'string') return 0;
				const match = value.match(/[\d,.]+/);
				if (!match) return 0;
				return parseFloat(match[0].replace(/,/g, '')) || 0;
			};

			const getMaxFromRange = (range: string): number => {
				if (typeof range !== 'string') return 0;
				const matches = range.match(/[\d,.]+/g);
				if (!matches) return 0;
				const numbers = matches.map(m => parseFloat(m.replace(/,/g, ''))).filter(n => !isNaN(n));
				return numbers.length > 0 ? Math.max(...numbers) : 0;
			};

			const numericPurchasePrice = parseFloat(purchasePrice) || 0;
			const numericARV = parseCurrency(parsedJson.suggestedARV);
			const numericMaxRehab = getMaxFromRange(estimation.summary.totalEstimatedCost);

			const numericMAO = (numericARV * 0.70) - numericMaxRehab;

			const formattedMAO = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(numericMAO);
			const formattedPurchasePrice = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(numericPurchasePrice);

			parsedJson.suggestedMAO = formattedMAO;
			parsedJson.purchasePrice = formattedPurchasePrice;

			const fitsCriteria = numericPurchasePrice > 0 && numericMAO > 0 && numericPurchasePrice <= numericMAO;
			parsedJson.investorFit.fitsCriteria = fitsCriteria;

			const dealVerdict = fitsCriteria
				? 'Based on the 70% rule, the purchase price is at or below the Maximum Allowable Offer. This indicates a potentially strong investment opportunity.'
				: `Warning: Based on the 70% rule, the Maximum Allowable Offer (MAO) for this property is ${formattedMAO}. The current purchase price of ${formattedPurchasePrice} is significantly higher than this target. For this deal to be profitable under standard investor criteria, the property would need to be acquired at or below the MAO.`;

			parsedJson.investorFit.analysis = `${dealVerdict}\n\n**AI Analysis:**\n${parsedJson.investorFit.analysis}`;

			// --- End of Application-Side Business Logic ---

			const groundingChunks = result.candidates?.[0]?.groundingMetadata?.groundingChunks ?? [];
			parsedJson.groundingSources = groundingChunks
				.filter((chunk: GroundingChunk) => chunk.web?.uri && chunk.web?.title)
				.map((chunk: GroundingChunk) => ({
					uri: chunk.web.uri,
					title: chunk.web.title,
				}));

			return parsedJson as InvestmentAnalysis;
		} else {
			throw new Error("Could not find JSON in the model's response for investment analysis.");
		}
	} catch (e) {
		console.error("Failed to parse investment analysis JSON:", e, "Raw response:", result.text);
		throw new Error("Failed to get a valid investment analysis from the AI.");
	}
};
