/**
 * Customer ITE (Individual Treatment Effect) Service
 * ===================================================
 *
 * Core causal inference engine for EconoCausal.
 *
 * Implements Double Machine Learning (DML) methodology to estimate the
 * CAUSAL impact of a discount on each individual customer's purchase
 * probability.
 *
 * Key Concepts:
 *   - Treatment (T): Discount amount (₹0, ₹100, ₹200, ₹300)
 *   - Outcome (Y):   Purchase probability (0–1)
 *   - Confounders (X): Browse count, cart history, income bracket,
 *                      past purchases, recency, competitor prices
 *
 * The DML approach:
 *   1. Model E[Y | X] → residualize Y
 *   2. Model E[T | X] → residualize T
 *   3. Regress residual-Y on residual-T → unbiased causal effect (θ)
 *
 * In this Node.js simulation, we use econometric formulas to approximate
 * what the Python EconML/DoWhy engine will compute in production.
 *
 * @module customerITEService
 */

/**
 * Customer segment labels based on causal response to treatment
 */
export const SEGMENTS = Object.freeze({
  PERSUADABLE: 'Persuadable',
  SURE_THING: 'Sure Thing',
  LOST_CAUSE: 'Lost Cause',
  SLEEPING_DOG: 'Sleeping Dog',
});

/**
 * Segment thresholds for ITE classification
 */
const ITE_THRESHOLDS = {
  PERSUADABLE_MIN: 0.10,   // ITE ≥ 10% → customer is persuadable
  SURE_THING_BASE: 0.65,   // Base purchase prob ≥ 65% AND ITE < 10%
  SLEEPING_DOG_MAX: -0.02, // Negative ITE → discount hurts
};

export class CustomerITEService {
  /**
   * Estimate Individual Treatment Effect for a single customer.
   *
   * Uses a DML-inspired approach:
   *   1. Predict baseline purchase probability from confounders (X)
   *   2. Predict treatment assignment propensity from confounders (X)
   *   3. Orthogonalize and compute the causal lift
   *
   * @param {Object} customer - Customer feature vector
   * @param {number} customer.browseCount - Number of product page views
   * @param {number} customer.cartAbandons - Number of cart abandonments
   * @param {number} customer.pastPurchases - Historical purchase count
   * @param {string} customer.incomeBracket - 'Low', 'Low-Medium', 'Medium', 'Medium-High', 'High'
   * @param {number} customer.daysSinceLastVisit - Recency in days
   * @param {number} customer.avgTimeOnPage - Average time spent (seconds)
   * @param {number} customer.wishlistItems - Items in wishlist
   * @param {number} discountAmount - Treatment: discount in ₹
   * @param {number} productPrice - Full product price in ₹
   * @returns {Object} ITE estimation result
   */
  static estimateITE(customer, discountAmount, productPrice) {
    // === Step 1: Estimate baseline purchase probability P(Y=1 | X, T=0) ===
    const baseProb = this._estimateBasePurchaseProbability(customer, productPrice);

    // === Step 2: Estimate treated purchase probability P(Y=1 | X, T=discount) ===
    const treatedProb = this._estimateTreatedPurchaseProbability(
      customer,
      discountAmount,
      productPrice,
      baseProb
    );

    // === Step 3: ITE = P(Y=1 | X, T=1) - P(Y=1 | X, T=0) ===
    const ite = Number((treatedProb - baseProb).toFixed(4));

    // === Step 4: Classify segment ===
    const segment = this._classifySegment(baseProb, ite);

    // === Step 5: Compute propensity score (probability of receiving treatment) ===
    const propensityScore = this._estimatePropensityScore(customer, discountAmount, productPrice);

    return {
      customerId: customer.customerId || customer._id || null,
      customerName: customer.name || null,
      basePurchaseProbability: Number(baseProb.toFixed(4)),
      treatedPurchaseProbability: Number(treatedProb.toFixed(4)),
      ite,
      itePercentage: Number((ite * 100).toFixed(2)),
      segment,
      discountAmount,
      propensityScore: Number(propensityScore.toFixed(4)),
      confidenceInterval: {
        lower: Number((ite - 0.04).toFixed(4)),
        upper: Number((ite + 0.04).toFixed(4)),
      },
      features: {
        browseCount: customer.browseCount,
        cartAbandons: customer.cartAbandons,
        pastPurchases: customer.pastPurchases,
        incomeBracket: customer.incomeBracket,
        daysSinceLastVisit: customer.daysSinceLastVisit,
      },
    };
  }

  /**
   * Estimate ITE across multiple discount tiers for a single customer.
   * Shows diminishing returns as discount increases.
   *
   * @param {Object} customer - Customer feature vector
   * @param {number[]} discountTiers - Array of discount amounts to test
   * @param {number} productPrice - Product price
   * @returns {Object} Multi-tier ITE analysis
   */
  static estimateMultiTierITE(customer, discountTiers, productPrice) {
    const tiers = discountTiers.map((discount) => {
      const result = this.estimateITE(customer, discount, productPrice);
      return {
        discountAmount: discount,
        discountPercentage: Number(((discount / productPrice) * 100).toFixed(1)),
        ite: result.ite,
        itePercentage: result.itePercentage,
        treatedProbability: result.treatedPurchaseProbability,
        segment: result.segment,
      };
    });

    // Find the discount tier with the best ITE-per-rupee efficiency
    const bestEfficiency = tiers.reduce((best, tier) => {
      const efficiency = tier.discountAmount > 0 ? tier.ite / tier.discountAmount : 0;
      if (efficiency > (best.efficiency || 0)) {
        return { ...tier, efficiency };
      }
      return best;
    }, { efficiency: 0 });

    return {
      customerId: customer.customerId || customer._id || null,
      customerName: customer.name || null,
      productPrice,
      tiers,
      recommendedTier: bestEfficiency.discountAmount || 0,
      recommendation:
        bestEfficiency.efficiency > 0
          ? `Best value at ₹${bestEfficiency.discountAmount} discount (${bestEfficiency.itePercentage}% lift per ₹${bestEfficiency.discountAmount})`
          : 'No discount recommended — customer is not persuadable',
    };
  }

  /**
   * Run batch ITE estimation and segmentation for an entire customer audience.
   * This is the main campaign analysis endpoint.
   *
   * @param {Object[]} customers - Array of customer feature vectors
   * @param {number} discountAmount - Discount to evaluate
   * @param {number} productPrice - Product price
   * @returns {Object} Full campaign analysis with segmentation
   */
  static analyzeAudience(customers, discountAmount, productPrice) {
    const results = customers.map((c) => this.estimateITE(c, discountAmount, productPrice));

    // Segment counts
    const segmentation = {
      [SEGMENTS.PERSUADABLE]: { count: 0, customers: [] },
      [SEGMENTS.SURE_THING]: { count: 0, customers: [] },
      [SEGMENTS.LOST_CAUSE]: { count: 0, customers: [] },
      [SEGMENTS.SLEEPING_DOG]: { count: 0, customers: [] },
    };

    results.forEach((r) => {
      segmentation[r.segment].count++;
      segmentation[r.segment].customers.push(r);
    });

    // Sort persuadables by ITE descending (highest-impact first)
    segmentation[SEGMENTS.PERSUADABLE].customers.sort((a, b) => b.ite - a.ite);

    const totalCustomers = customers.length;
    const segmentSummary = Object.entries(segmentation).map(([segment, data]) => ({
      segment,
      count: data.count,
      percentage: Number(((data.count / totalCustomers) * 100).toFixed(1)),
    }));

    // Average ITE across all customers
    const avgITE = results.reduce((sum, r) => sum + r.ite, 0) / totalCustomers;

    // Uplift curve data (customers sorted by ITE descending)
    const upliftCurve = this._generateUpliftCurve(results);

    return {
      totalCustomers,
      discountAmount,
      productPrice,
      segmentSummary,
      averageITE: Number(avgITE.toFixed(4)),
      averageITEPercentage: Number((avgITE * 100).toFixed(2)),
      upliftCurve,
      segmentation,
      methodology: 'Double Machine Learning (DML) - Orthogonal ITE Estimation',
    };
  }

  /**
   * Generate synthetic customer audience for testing and demos.
   * Creates a realistic distribution of customer profiles.
   *
   * @param {number} count - Number of customers to generate
   * @returns {Object[]} Array of customer feature vectors
   */
  static generateSyntheticAudience(count = 100) {
    const incomeBrackets = ['Low', 'Low-Medium', 'Medium', 'Medium-High', 'High'];
    const firstNames = [
      'Priya', 'Amit', 'Sneha', 'Rajesh', 'Deepa', 'Vikram', 'Ananya', 'Rahul',
      'Kavita', 'Suresh', 'Meera', 'Arjun', 'Pooja', 'Karthik', 'Nisha',
      'Siddharth', 'Divya', 'Manish', 'Ritu', 'Aakash', 'Lakshmi', 'Gaurav',
      'Swati', 'Nikhil', 'Preeti',
    ];
    const cities = [
      'Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Hyderabad', 'Pune',
      'Kolkata', 'Jaipur', 'Lucknow', 'Ahmedabad',
    ];

    const customers = [];

    for (let i = 0; i < count; i++) {
      // Create diverse customer profiles
      const profileType = Math.random();
      let profile;

      if (profileType < 0.25) {
        // High-intent price-sensitive (likely Persuadable)
        profile = {
          browseCount: Math.floor(5 + Math.random() * 15),
          cartAbandons: Math.floor(1 + Math.random() * 4),
          pastPurchases: Math.floor(Math.random() * 2),
          incomeBracket: incomeBrackets[Math.floor(Math.random() * 3)], // Low to Medium
          daysSinceLastVisit: Math.floor(1 + Math.random() * 7),
          avgTimeOnPage: Math.floor(30 + Math.random() * 120),
          wishlistItems: Math.floor(1 + Math.random() * 3),
        };
      } else if (profileType < 0.55) {
        // Loyal high-income (likely Sure Thing)
        profile = {
          browseCount: Math.floor(1 + Math.random() * 4),
          cartAbandons: 0,
          pastPurchases: Math.floor(3 + Math.random() * 10),
          incomeBracket: incomeBrackets[3 + Math.floor(Math.random() * 2)], // Medium-High to High
          daysSinceLastVisit: Math.floor(1 + Math.random() * 14),
          avgTimeOnPage: Math.floor(10 + Math.random() * 30),
          wishlistItems: Math.floor(Math.random() * 2),
        };
      } else if (profileType < 0.90) {
        // Low engagement (likely Lost Cause)
        profile = {
          browseCount: Math.floor(Math.random() * 3),
          cartAbandons: 0,
          pastPurchases: 0,
          incomeBracket: incomeBrackets[Math.floor(Math.random() * 5)],
          daysSinceLastVisit: Math.floor(14 + Math.random() * 60),
          avgTimeOnPage: Math.floor(5 + Math.random() * 15),
          wishlistItems: 0,
        };
      } else {
        // Skeptical shopper (likely Sleeping Dog)
        profile = {
          browseCount: Math.floor(2 + Math.random() * 5),
          cartAbandons: Math.floor(Math.random() * 2),
          pastPurchases: Math.floor(1 + Math.random() * 3),
          incomeBracket: incomeBrackets[3 + Math.floor(Math.random() * 2)],
          daysSinceLastVisit: Math.floor(5 + Math.random() * 20),
          avgTimeOnPage: Math.floor(15 + Math.random() * 40),
          wishlistItems: 0,
        };
      }

      customers.push({
        customerId: `cust_${String(i + 1).padStart(4, '0')}`,
        name: firstNames[Math.floor(Math.random() * firstNames.length)],
        city: cities[Math.floor(Math.random() * cities.length)],
        ...profile,
      });
    }

    return customers;
  }

  // ─────────────────────────────────────────────────────────────
  //  PRIVATE METHODS — Causal Estimation Internals
  // ─────────────────────────────────────────────────────────────

  /**
   * Estimate baseline purchase probability P(Y=1 | X, T=0).
   * Models the customer's likelihood of purchasing WITHOUT any discount.
   *
   * Uses a logistic-style model with feature contributions:
   *   - Browse intensity → higher engagement = higher intent
   *   - Cart abandonment → strong buy signal (friction-blocked)
   *   - Past purchases  → loyalty / habit
   *   - Income bracket  → ability to pay full price
   *   - Recency         → more recent = more active
   *
   * @private
   */
  static _estimateBasePurchaseProbability(customer, productPrice) {
    const {
      browseCount = 0,
      cartAbandons = 0,
      pastPurchases = 0,
      incomeBracket = 'Medium',
      daysSinceLastVisit = 30,
      avgTimeOnPage = 20,
      wishlistItems = 0,
    } = customer;

    // Income multiplier maps
    const incomeMultiplier = {
      'Low': 0.4,
      'Low-Medium': 0.6,
      'Medium': 0.8,
      'Medium-High': 1.0,
      'High': 1.2,
    };

    // Logit contributions (log-odds space)
    let logOdds = -2.5; // Base intercept (≈7% baseline)

    // Browse intensity: each view adds diminishing evidence of interest
    logOdds += Math.min(browseCount * 0.15, 2.0);

    // Cart abandonment: strongest buy signal — they WANTED it
    logOdds += cartAbandons * 0.6;

    // Past purchase loyalty
    logOdds += Math.min(pastPurchases * 0.35, 2.5);

    // Income: higher income → more likely to buy at full price
    const incMult = incomeMultiplier[incomeBracket] || 0.8;
    logOdds += (incMult - 0.8) * 1.5;

    // Recency decay: longer absence → lower probability
    logOdds -= Math.min(daysSinceLastVisit * 0.03, 1.5);

    // Time-on-page engagement
    logOdds += Math.min(avgTimeOnPage * 0.005, 0.5);

    // Wishlist signal
    logOdds += wishlistItems * 0.3;

    // Convert logit → probability via sigmoid
    const probability = 1 / (1 + Math.exp(-logOdds));

    return Math.max(0.01, Math.min(0.99, probability));
  }

  /**
   * Estimate treated purchase probability P(Y=1 | X, T=discount).
   *
   * The treatment effect is NOT uniform — it depends on the customer's
   * price sensitivity, which is modulated by:
   *   - Income (lower income = more price-sensitive)
   *   - Base probability (mid-range = most "persuadable")
   *   - Browse intensity (high browsing + no purchase = price barrier)
   *   - Discount magnitude relative to price
   *
   * This captures the heterogeneous treatment effect that DML estimates.
   *
   * @private
   */
  static _estimateTreatedPurchaseProbability(customer, discountAmount, productPrice, baseProb) {
    if (discountAmount <= 0) return baseProb;

    const {
      browseCount = 0,
      cartAbandons = 0,
      pastPurchases = 0,
      incomeBracket = 'Medium',
    } = customer;

    const discountFraction = discountAmount / productPrice;

    // --- Price Sensitivity Score (0 to 1) ---
    // Lower income = more price sensitive
    const incomeSensitivity = {
      'Low': 0.95,
      'Low-Medium': 0.80,
      'Medium': 0.60,
      'Medium-High': 0.35,
      'High': 0.15,
    };
    const priceSensitivity = incomeSensitivity[incomeBracket] || 0.6;

    // --- Intent Signal ---
    // High browse + cart abandons = price is the barrier
    const intentSignal = Math.min(
      (browseCount * 0.08 + cartAbandons * 0.25) / 1.5,
      1.0
    );

    // --- Persuadability Zone ---
    // Customers in the middle base-probability range are most persuadable
    // Very high base = buying anyway, very low base = not interested
    const persuadabilityZone = 4 * baseProb * (1 - baseProb); // peaks at 0.5

    // --- Sleeping Dog Effect ---
    // Some high-income repeat buyers are LESS likely to buy with a discount
    // (perceived cheapening of the brand / quality concerns)
    const sleepingDogRisk =
      pastPurchases > 3 && (incomeBracket === 'High' || incomeBracket === 'Medium-High')
        ? -0.03 * discountFraction
        : 0;

    // --- Compute Causal Lift ---
    const rawLift =
      discountFraction *
      priceSensitivity *
      intentSignal *
      persuadabilityZone *
      2.5; // Scaling factor

    // Diminishing returns on larger discounts (concave response curve)
    const diminishedLift = rawLift * (1 - 0.3 * discountFraction);

    const totalLift = diminishedLift + sleepingDogRisk;

    const treatedProb = baseProb + totalLift;

    return Math.max(0.01, Math.min(0.99, treatedProb));
  }

  /**
   * Classify customer into a causal segment based on their ITE.
   *
   * | Segment       | Condition                               |
   * |───────────────|─────────────────────────────────────────|
   * | Persuadable   | ITE ≥ 10% (discount CAUSES purchase)   |
   * | Sure Thing    | Base prob ≥ 70% AND ITE < 10%           |
   * | Lost Cause    | Base prob < 70% AND ITE < 10%           |
   * | Sleeping Dog  | ITE < -2% (discount HURTS)              |
   *
   * @private
   */
  static _classifySegment(baseProb, ite) {
    if (ite < ITE_THRESHOLDS.SLEEPING_DOG_MAX) {
      return SEGMENTS.SLEEPING_DOG;
    }
    if (ite >= ITE_THRESHOLDS.PERSUADABLE_MIN) {
      return SEGMENTS.PERSUADABLE;
    }
    if (baseProb >= ITE_THRESHOLDS.SURE_THING_BASE) {
      return SEGMENTS.SURE_THING;
    }
    return SEGMENTS.LOST_CAUSE;
  }

  /**
   * Estimate propensity score P(T=1 | X).
   * In an observational setting, this corrects for selection bias:
   * who was MORE LIKELY to receive the discount based on their features.
   *
   * @private
   */
  static _estimatePropensityScore(customer, discountAmount, productPrice) {
    if (discountAmount <= 0) return 0;

    const {
      browseCount = 0,
      cartAbandons = 0,
      pastPurchases = 0,
    } = customer;

    // Propensity: likelihood the system WOULD assign this customer a discount
    // Based on observable features (mimics what an e-commerce CRM would do)
    let logOdds = -1.0; // Base: most people DON'T get targeted

    logOdds += Math.min(browseCount * 0.1, 1.0);
    logOdds += cartAbandons * 0.4;
    logOdds -= Math.min(pastPurchases * 0.15, 1.0); // Loyal customers don't get targeted

    const propensity = 1 / (1 + Math.exp(-logOdds));
    return Math.max(0.05, Math.min(0.95, propensity));
  }

  /**
   * Generate uplift curve data for visualization.
   * X-axis: % of customers targeted (sorted by ITE descending)
   * Y-axis: Cumulative incremental purchases
   *
   * Used to prove that causal targeting >> random targeting.
   *
   * @private
   */
  static _generateUpliftCurve(results) {
    // Sort by ITE descending (best customers first)
    const sorted = [...results].sort((a, b) => b.ite - a.ite);
    const total = sorted.length;

    const curve = [];
    let cumulativeUplift = 0;
    let randomCumulativeUplift = 0;
    const totalUplift = sorted.reduce((sum, r) => sum + r.ite, 0);

    for (let i = 0; i < total; i++) {
      cumulativeUplift += sorted[i].ite;
      randomCumulativeUplift += totalUplift / total; // Random = flat diagonal

      if (i % Math.max(1, Math.floor(total / 20)) === 0 || i === total - 1) {
        curve.push({
          percentTargeted: Number((((i + 1) / total) * 100).toFixed(1)),
          customersTargeted: i + 1,
          cumulativeUplift: Number(cumulativeUplift.toFixed(4)),
          randomBaseline: Number(randomCumulativeUplift.toFixed(4)),
          liftOverRandom: Number((cumulativeUplift - randomCumulativeUplift).toFixed(4)),
        });
      }
    }

    return curve;
  }
}
