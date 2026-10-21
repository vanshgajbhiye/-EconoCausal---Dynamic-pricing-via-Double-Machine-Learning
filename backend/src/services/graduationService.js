/**
 * Graduation Service
 * ==================
 *
 * Tracks customer lifecycle transitions in the EconoCausal system.
 *
 * Key Innovation: Customers don't stay "Persuadable" forever.
 * After repeated discount-driven purchases, they develop brand loyalty
 * and naturally GRADUATE to "Sure Thing" status — at which point
 * discounts are tapered off and the freed budget is reallocated
 * to new Persuadables.
 *
 * Lifecycle Flow:
 *   Persuadable (needs discount)
 *       │
 *       ├── Purchase #1: ITE = +54% → Full discount
 *       ├── Purchase #2: ITE = +41% → Full discount
 *       ├── Purchase #3: ITE = +22% → Tapering (reduced discount)
 *       ├── Purchase #4: ITE = +8%  → Tapering (minimal discount)
 *       └── Purchase #5: ITE = +2%  → 🎓 GRADUATED (no discount needed)
 *
 * The service:
 *   1. Tracks purchase history per customer
 *   2. Detects ITE decay (natural loyalty building)
 *   3. Recommends discount tapering schedule
 *   4. Flags graduated customers
 *   5. Calculates budget freed by graduations
 *
 * @module graduationService
 */

import { CustomerITEService, SEGMENTS } from './customerITEService.js';

/**
 * Graduation status labels
 */
export const GRADUATION_STATUS = Object.freeze({
  ACTIVE_PERSUADABLE: 'Active Persuadable',
  TAPERING: 'Tapering',
  NEAR_GRADUATION: 'Near Graduation',
  GRADUATED: 'Graduated',
  SURE_THING: 'Already Sure Thing',
  NOT_APPLICABLE: 'Not Applicable',
});

/**
 * Thresholds for graduation lifecycle
 */
const GRADUATION_THRESHOLDS = {
  TAPER_START_PURCHASES: 3,       // Start tapering after 3 discount-driven purchases
  GRADUATION_PURCHASES: 5,        // Graduate after 5 discount-driven purchases
  ITE_GRADUATED: 0.05,            // ITE below 5% = graduated
  ITE_NEAR_GRADUATION: 0.10,      // ITE 5-10% = near graduation
  TAPER_DISCOUNT_MULTIPLIERS: [1.0, 1.0, 0.75, 0.50, 0.25, 0], // Per-purchase multiplier
};

export class GraduationService {
  /**
   * Evaluate a single customer's graduation status.
   *
   * @param {Object} params
   * @param {Object} params.customer - Customer feature vector
   * @param {number} params.discountDrivenPurchases - Number of purchases made with discounts
   * @param {number} params.currentDiscountAmount - Current discount being given
   * @param {number} params.productPrice - Product price
   * @returns {Object} Graduation assessment
   */
  static evaluateGraduation({ customer, discountDrivenPurchases, currentDiscountAmount, productPrice }) {
    // Simulate how ITE decays as purchases increase
    // Each purchase builds loyalty → base probability increases → ITE drops
    const iteHistory = this._simulateITEDecay(customer, currentDiscountAmount, productPrice, discountDrivenPurchases);

    const currentITE = iteHistory[iteHistory.length - 1];
    const initialITE = iteHistory[0];
    const iteDecayRate = discountDrivenPurchases > 0
      ? Number(((1 - currentITE.ite / Math.max(initialITE.ite, 0.01)) * 100).toFixed(1))
      : 0;

    // Determine graduation status
    let status;
    let recommendedDiscount;

    if (currentITE.ite < GRADUATION_THRESHOLDS.ITE_GRADUATED) {
      status = GRADUATION_STATUS.GRADUATED;
      recommendedDiscount = 0;
    } else if (currentITE.ite < GRADUATION_THRESHOLDS.ITE_NEAR_GRADUATION) {
      status = GRADUATION_STATUS.NEAR_GRADUATION;
      recommendedDiscount = Math.round(currentDiscountAmount * 0.25);
    } else if (discountDrivenPurchases >= GRADUATION_THRESHOLDS.TAPER_START_PURCHASES) {
      status = GRADUATION_STATUS.TAPERING;
      const multiplierIndex = Math.min(
        discountDrivenPurchases,
        GRADUATION_THRESHOLDS.TAPER_DISCOUNT_MULTIPLIERS.length - 1
      );
      recommendedDiscount = Math.round(
        currentDiscountAmount * GRADUATION_THRESHOLDS.TAPER_DISCOUNT_MULTIPLIERS[multiplierIndex]
      );
    } else {
      status = GRADUATION_STATUS.ACTIVE_PERSUADABLE;
      recommendedDiscount = currentDiscountAmount;
    }

    const budgetSaved = currentDiscountAmount - recommendedDiscount;

    return {
      customerId: customer.customerId || customer._id || null,
      customerName: customer.name || null,
      status,
      discountDrivenPurchases,
      currentDiscount: currentDiscountAmount,
      recommendedDiscount,
      budgetSavedPerTransaction: budgetSaved,
      iteDecayPercentage: iteDecayRate,
      currentITE: currentITE.ite,
      currentITEPercentage: currentITE.itePercentage,
      initialITE: initialITE.ite,
      initialITEPercentage: initialITE.itePercentage,
      purchasesToGraduation: Math.max(0, GRADUATION_THRESHOLDS.GRADUATION_PURCHASES - discountDrivenPurchases),
      iteHistory,
      taperSchedule: this._generateTaperSchedule(currentDiscountAmount, discountDrivenPurchases),
    };
  }

  /**
   * Evaluate graduation status for an entire cohort.
   * Generates a summary report of the graduating class.
   *
   * @param {Object[]} customerRecords - Array of {customer, discountDrivenPurchases, currentDiscountAmount}
   * @param {number} productPrice - Product price
   * @returns {Object} Cohort graduation report
   */
  static evaluateCohort(customerRecords, productPrice) {
    const assessments = customerRecords.map((record) =>
      this.evaluateGraduation({
        customer: record.customer,
        discountDrivenPurchases: record.discountDrivenPurchases,
        currentDiscountAmount: record.currentDiscountAmount,
        productPrice,
      })
    );

    // Group by status
    const statusGroups = {};
    Object.values(GRADUATION_STATUS).forEach((s) => {
      statusGroups[s] = { count: 0, customers: [], totalBudgetSaved: 0 };
    });

    assessments.forEach((a) => {
      const group = statusGroups[a.status];
      if (group) {
        group.count++;
        group.customers.push(a);
        group.totalBudgetSaved += a.budgetSavedPerTransaction;
      }
    });

    const totalBudgetFreed = assessments.reduce((sum, a) => sum + a.budgetSavedPerTransaction, 0);
    const graduatedCount = statusGroups[GRADUATION_STATUS.GRADUATED].count;
    const taperingCount = statusGroups[GRADUATION_STATUS.TAPERING].count;
    const nearGradCount = statusGroups[GRADUATION_STATUS.NEAR_GRADUATION].count;

    return {
      totalCustomersEvaluated: customerRecords.length,
      statusSummary: Object.entries(statusGroups).map(([status, data]) => ({
        status,
        count: data.count,
        percentage: Number(((data.count / customerRecords.length) * 100).toFixed(1)),
        budgetSaved: Number(data.totalBudgetSaved.toFixed(2)),
      })),
      totalMonthlyBudgetFreed: Number(totalBudgetFreed.toFixed(2)),
      highlights: {
        graduated: graduatedCount,
        tapering: taperingCount,
        nearGraduation: nearGradCount,
        activePersuadables: statusGroups[GRADUATION_STATUS.ACTIVE_PERSUADABLE].count,
      },
      narrative: this._generateNarrative(graduatedCount, totalBudgetFreed, taperingCount),
      assessments,
    };
  }

  /**
   * Generate demo graduation data for the dashboard.
   * Simulates a cohort at various stages of the graduation lifecycle.
   *
   * @param {number} cohortSize - Number of customers in the demo cohort
   * @param {number} discountAmount - Base discount amount
   * @param {number} productPrice - Product price
   * @returns {Object} Demo cohort data
   */
  static generateDemoCohort(cohortSize = 50, discountAmount = 200, productPrice = 2499) {
    const customers = CustomerITEService.generateSyntheticAudience(cohortSize);

    // Assign varying purchase histories to simulate lifecycle stages
    const customerRecords = customers.map((customer, index) => {
      // Distribute across lifecycle stages
      let discountDrivenPurchases;
      const stage = index % 10;

      if (stage < 3) {
        discountDrivenPurchases = Math.floor(Math.random() * 2); // New (0-1 purchases)
      } else if (stage < 6) {
        discountDrivenPurchases = 2 + Math.floor(Math.random() * 2); // Active (2-3)
      } else if (stage < 8) {
        discountDrivenPurchases = 3 + Math.floor(Math.random() * 2); // Tapering (3-4)
      } else if (stage < 9) {
        discountDrivenPurchases = 4 + Math.floor(Math.random() * 2); // Near graduation (4-5)
      } else {
        discountDrivenPurchases = 5 + Math.floor(Math.random() * 3); // Graduated (5+)
      }

      return {
        customer,
        discountDrivenPurchases,
        currentDiscountAmount: discountAmount,
      };
    });

    return this.evaluateCohort(customerRecords, productPrice);
  }

  // ─────────────────────────────────────────────────────────────
  //  PRIVATE METHODS
  // ─────────────────────────────────────────────────────────────

  /**
   * Simulate how a customer's ITE decays over repeated discount-driven purchases.
   *
   * Each purchase builds loyalty:
   *   - Past purchases increase → base probability increases
   *   - Higher base prob → customer enters "Sure Thing" zone
   *   - ITE naturally drops → discount becomes unnecessary
   *
   * @private
   */
  static _simulateITEDecay(customer, discountAmount, productPrice, maxPurchases) {
    const history = [];
    const steps = Math.max(maxPurchases + 1, 6); // Show at least 6 points

    for (let purchase = 0; purchase < steps; purchase++) {
      // Simulate increasing loyalty with each purchase
      const evolvedCustomer = {
        ...customer,
        pastPurchases: (customer.pastPurchases || 0) + purchase,
        // Cart abandons decrease as they become regular buyers
        cartAbandons: Math.max(0, (customer.cartAbandons || 0) - Math.floor(purchase / 2)),
        // Browse count normalizes
        browseCount: Math.max(1, (customer.browseCount || 1) - Math.floor(purchase / 3)),
      };

      // Apply taper multiplier to discount
      const multiplierIndex = Math.min(purchase, GRADUATION_THRESHOLDS.TAPER_DISCOUNT_MULTIPLIERS.length - 1);
      const taperedDiscount = Math.round(
        discountAmount * GRADUATION_THRESHOLDS.TAPER_DISCOUNT_MULTIPLIERS[multiplierIndex]
      );

      const result = CustomerITEService.estimateITE(evolvedCustomer, taperedDiscount, productPrice);

      history.push({
        purchaseNumber: purchase + 1,
        discountGiven: taperedDiscount,
        ite: result.ite,
        itePercentage: result.itePercentage,
        baseProbability: result.basePurchaseProbability,
        treatedProbability: result.treatedPurchaseProbability,
        segment: result.segment,
      });
    }

    return history;
  }

  /**
   * Generate the recommended discount taper schedule.
   *
   * @private
   */
  static _generateTaperSchedule(baseDiscount, currentPurchase) {
    return GRADUATION_THRESHOLDS.TAPER_DISCOUNT_MULTIPLIERS.map((mult, index) => {
      const discount = Math.round(baseDiscount * mult);
      return {
        purchaseNumber: index + 1,
        discountMultiplier: mult,
        discountAmount: discount,
        status: index < currentPurchase
          ? '✅ Completed'
          : index === currentPurchase
            ? '🔄 Current'
            : '⏳ Upcoming',
      };
    });
  }

  /**
   * Generate a human-readable narrative for the graduation report.
   *
   * @private
   */
  static _generateNarrative(graduatedCount, budgetFreed, taperingCount) {
    const parts = [];

    if (graduatedCount > 0) {
      parts.push(
        `${graduatedCount} customer${graduatedCount > 1 ? 's' : ''} graduated this period — they now buy without discounts.`
      );
    }

    if (budgetFreed > 0) {
      parts.push(
        `₹${budgetFreed.toFixed(0)} monthly budget freed and available for reallocation to new Persuadables.`
      );
    }

    if (taperingCount > 0) {
      parts.push(
        `${taperingCount} customer${taperingCount > 1 ? 's are' : ' is'} in the tapering phase — discounts being gradually reduced.`
      );
    }

    if (parts.length === 0) {
      parts.push('No graduations detected yet. Continue monitoring customer ITE trends.');
    }

    return parts.join(' ');
  }
}
