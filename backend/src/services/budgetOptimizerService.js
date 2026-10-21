/**
 * Budget Optimizer Service
 * ========================
 *
 * Solves the budget-constrained discount allocation problem:
 *   "Given a fixed marketing budget, how should we distribute
 *    personalized discounts to maximize total incremental purchases?"
 *
 * This is a variant of the Knapsack problem:
 *   - Each Persuadable customer has multiple discount tiers
 *   - Each tier has a "weight" (cost = discount amount) and
 *     "value" (ITE = incremental purchase probability)
 *   - Maximize total ITE subject to sum(discounts) ≤ budget
 *
 * Algorithm:
 *   1. For each Persuadable, compute ITE at every discount tier
 *   2. Compute efficiency = ITE / discount for each tier
 *   3. Greedy allocation: assign highest-efficiency tiers first
 *   4. If budget remains, upgrade customers to next-best tier
 *
 * In production, this would use SciPy's linear programming solver.
 * Here we use an efficient greedy approximation.
 *
 * @module budgetOptimizerService
 */

import { CustomerITEService, SEGMENTS } from './customerITEService.js';

export class BudgetOptimizerService {
  /**
   * Run the full budget-constrained discount allocation.
   *
   * @param {Object} params
   * @param {Object[]} params.customers - Array of customer feature vectors
   * @param {number} params.budget - Total marketing budget in ₹
   * @param {number[]} params.discountTiers - Discount amounts to consider (e.g., [100, 200, 300])
   * @param {number} params.productPrice - Product price in ₹
   * @returns {Object} Optimal allocation plan
   */
  static optimize({ customers, budget, discountTiers, productPrice }) {
    if (!customers || customers.length === 0) {
      throw new Error('Customer list is required for optimization');
    }
    if (!budget || budget <= 0) {
      throw new Error('A positive budget is required');
    }
    if (!discountTiers || discountTiers.length === 0) {
      throw new Error('At least one discount tier is required');
    }

    // Sort tiers ascending
    const sortedTiers = [...discountTiers].sort((a, b) => a - b);

    // Step 1: Compute multi-tier ITE for every customer
    const customerAnalyses = customers.map((customer) => {
      const multiTier = CustomerITEService.estimateMultiTierITE(customer, sortedTiers, productPrice);
      const baseResult = CustomerITEService.estimateITE(customer, 0, productPrice);

      // Determine segment using the BEST tier (highest discount), not discount=0.
      // A customer is Persuadable if any discount tier pushes their ITE ≥ threshold.
      const bestTierResult = CustomerITEService.estimateITE(
        customer,
        sortedTiers[sortedTiers.length - 1],
        productPrice
      );

      return {
        customerId: customer.customerId || customer._id,
        customerName: customer.name || null,
        city: customer.city || null,
        baseProbability: baseResult.basePurchaseProbability,
        segment: bestTierResult.segment,
        tiers: multiTier.tiers.map((tier) => ({
          ...tier,
          // Marginal efficiency: how much ITE do we get per ₹ spent?
          efficiency: tier.discountAmount > 0
            ? Number((tier.ite / tier.discountAmount).toFixed(6))
            : 0,
        })),
      };
    });

    // Step 2: Filter to only Persuadable customers (only they should get discounts)
    const persuadables = customerAnalyses.filter(
      (ca) => ca.segment === SEGMENTS.PERSUADABLE
    );

    // Step 3: Build candidate allocations (each persuadable × each tier)
    const candidates = [];
    persuadables.forEach((customer) => {
      customer.tiers.forEach((tier) => {
        if (tier.discountAmount > 0 && tier.ite > 0) {
          candidates.push({
            customerId: customer.customerId,
            customerName: customer.customerName,
            city: customer.city,
            discountAmount: tier.discountAmount,
            ite: tier.ite,
            itePercentage: tier.itePercentage,
            efficiency: tier.efficiency,
            treatedProbability: tier.treatedProbability,
          });
        }
      });
    });

    // Step 4: Greedy allocation by efficiency (highest ITE-per-₹ first)
    candidates.sort((a, b) => b.efficiency - a.efficiency);

    let remainingBudget = budget;
    const allocations = new Map(); // customerId → best allocation

    for (const candidate of candidates) {
      // Skip if this customer already has a more efficient allocation
      const existing = allocations.get(candidate.customerId);
      if (existing) {
        // Only upgrade if budget allows AND the higher tier gives meaningful uplift
        const marginalCost = candidate.discountAmount - existing.discountAmount;
        const marginalITE = candidate.ite - existing.ite;

        if (marginalCost > 0 && marginalITE > 0.02 && marginalCost <= remainingBudget) {
          remainingBudget += existing.discountAmount; // Refund old
          remainingBudget -= candidate.discountAmount; // Charge new
          allocations.set(candidate.customerId, candidate);
        }
        continue;
      }

      // New allocation
      if (candidate.discountAmount <= remainingBudget) {
        remainingBudget -= candidate.discountAmount;
        allocations.set(candidate.customerId, candidate);
      }
    }

    // Step 5: Build the allocation matrix
    const allocationList = Array.from(allocations.values()).sort((a, b) => b.ite - a.ite);

    // Step 6: Compute summary statistics
    const totalSpent = budget - remainingBudget;
    const totalIncrementalPurchases = allocationList.reduce((sum, a) => sum + a.ite, 0);
    const avgDiscountPerCustomer =
      allocationList.length > 0
        ? totalSpent / allocationList.length
        : 0;

    // Step 7: Tier breakdown
    const tierBreakdown = sortedTiers.map((tier) => {
      const inTier = allocationList.filter((a) => a.discountAmount === tier);
      return {
        discountAmount: tier,
        customersAllocated: inTier.length,
        totalCost: inTier.length * tier,
        avgITE: inTier.length > 0
          ? Number((inTier.reduce((s, a) => s + a.ite, 0) / inTier.length).toFixed(4))
          : 0,
      };
    });

    // Step 8: Compare vs blanket and random strategies
    const roiComparison = this._compareStrategies({
      customers: customerAnalyses,
      allocationList,
      budget,
      totalSpent,
      productPrice,
      sortedTiers,
    });

    return {
      budget,
      totalSpent: Number(totalSpent.toFixed(2)),
      remainingBudget: Number(remainingBudget.toFixed(2)),
      utilizationPercentage: Number(((totalSpent / budget) * 100).toFixed(1)),
      totalCustomersAnalyzed: customers.length,
      persuadablesFound: persuadables.length,
      customersAllocated: allocationList.length,
      totalIncrementalPurchases: Number(totalIncrementalPurchases.toFixed(2)),
      avgDiscountPerCustomer: Number(avgDiscountPerCustomer.toFixed(2)),
      tierBreakdown,
      allocationMatrix: allocationList,
      roiComparison,
      methodology: 'Greedy Knapsack Optimization with DML-estimated ITEs',
    };
  }

  /**
   * Compare EconoCausal's optimal allocation against naive strategies.
   *
   * Strategies compared:
   *   1. Blanket: Give lowest-tier discount to everyone (capped by budget)
   *   2. Random: Give discounts to random customers
   *   3. EconoCausal: Give discounts only to Persuadables, optimized per-customer
   *
   * @private
   */
  static _compareStrategies({ customers, allocationList, budget, totalSpent, productPrice, sortedTiers }) {
    const lowestDiscount = sortedTiers[0];
    const totalCustomers = customers.length;

    // --- Blanket Strategy ---
    const blanketCustomersReached = Math.min(
      Math.floor(budget / lowestDiscount),
      totalCustomers
    );
    const blanketSpend = blanketCustomersReached * lowestDiscount;
    // Random selection means avg ITE across all customers at that tier
    const allITEs = customers.map((c) => {
      const tier = c.tiers.find((t) => t.discountAmount === lowestDiscount);
      return tier ? tier.ite : 0;
    });
    const blanketAvgITE = allITEs.reduce((s, v) => s + v, 0) / totalCustomers;
    const blanketIncrPurchases = blanketCustomersReached * blanketAvgITE;

    // --- Random Strategy ---
    // Randomly pick the same number of customers as EconoCausal allocated
    const randomCount = allocationList.length;
    const randomAvgITE = blanketAvgITE; // Same avg since random
    const randomIncrPurchases = randomCount * randomAvgITE;
    const randomSpend = randomCount * lowestDiscount;

    // --- EconoCausal Optimal ---
    const optimalIncrPurchases = allocationList.reduce((s, a) => s + a.ite, 0);

    // Revenue estimates (incremental units × price)
    const blanketRevenue = Number((blanketIncrPurchases * productPrice).toFixed(2));
    const randomRevenue = Number((randomIncrPurchases * productPrice).toFixed(2));
    const optimalRevenue = Number((optimalIncrPurchases * productPrice).toFixed(2));

    return {
      blanket: {
        strategy: `₹${lowestDiscount} discount to everyone`,
        budgetUsed: Number(blanketSpend.toFixed(2)),
        customersTargeted: blanketCustomersReached,
        incrementalPurchases: Number(blanketIncrPurchases.toFixed(2)),
        estimatedRevenue: blanketRevenue,
        roi: blanketSpend > 0 ? Number((blanketRevenue / blanketSpend).toFixed(1)) : 0,
      },
      random: {
        strategy: 'Random targeting',
        budgetUsed: Number(randomSpend.toFixed(2)),
        customersTargeted: randomCount,
        incrementalPurchases: Number(randomIncrPurchases.toFixed(2)),
        estimatedRevenue: randomRevenue,
        roi: randomSpend > 0 ? Number((randomRevenue / randomSpend).toFixed(1)) : 0,
      },
      optimal: {
        strategy: 'EconoCausal Optimal (DML + Budget Optimization)',
        budgetUsed: Number(totalSpent.toFixed(2)),
        customersTargeted: allocationList.length,
        incrementalPurchases: Number(optimalIncrPurchases.toFixed(2)),
        estimatedRevenue: optimalRevenue,
        roi: totalSpent > 0 ? Number((optimalRevenue / totalSpent).toFixed(1)) : 0,
      },
      savingsVsBlanket: Number((blanketSpend - totalSpent).toFixed(2)),
      revenueGainVsRandom: Number((optimalRevenue - randomRevenue).toFixed(2)),
      summary: optimalRevenue > randomRevenue
        ? `EconoCausal generates ₹${(optimalRevenue - randomRevenue).toFixed(0)} more revenue than random targeting and saves ₹${(blanketSpend - totalSpent).toFixed(0)} compared to blanket discounting.`
        : 'Analysis complete. Consider adjusting discount tiers or budget.',
    };
  }
}
