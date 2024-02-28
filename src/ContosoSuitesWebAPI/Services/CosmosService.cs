using System.Runtime.CompilerServices;
using Microsoft.Azure.Cosmos;
using Microsoft.Azure.Cosmos.Linq;

namespace ContosoSuitesWebAPI.Services;

public class CosmosService : ICosmosService
{
    private readonly CosmosClient _client;
    private Container container
    {
        get => _client.GetDatabase("ContosoSuites").GetContainer("Customers");
    }

    public CosmosService()
    {
        _client = new CosmosClient(
            connectionString: Environment.GetEnvironmentVariable("AZURE_COSMOS_DB_CONNECTION_STRING")!
        );
    }

    public async Task<IEnumerable<Customer>> GetCustomersByName(string name)
    {
        var queryable = container.GetItemLinqQueryable<Customer>();
        using FeedIterator<Customer> feed = queryable
            .Where(c => c.FullName == name)
            .ToFeedIterator<Customer>();
        return await ExecuteQuery(feed);
    }

    public async Task<IEnumerable<Customer>> GetCustomersByLoyaltyTier(string loyaltyTier)
    {
        LoyaltyTier lt = Enum.Parse<LoyaltyTier>(loyaltyTier);
        var queryable = container.GetItemLinqQueryable<Customer>();
        using FeedIterator<Customer> feed = queryable
            .Where(c => c.LoyaltyTier.ToString() == loyaltyTier)
            .ToFeedIterator<Customer>();
        return await ExecuteQuery(feed);
    }

    public async Task<IEnumerable<Customer>> GetCustomersWithStaysAfterDate(DateTime dt)
    {
        var queryable = container.GetItemLinqQueryable<Customer>();
        using FeedIterator<Customer> feed = queryable
            .Where(c => c.DateOfMostRecentStay > dt)
            .ToFeedIterator<Customer>();
        return await ExecuteQuery(feed);
    }

    private async Task<IEnumerable<Customer>> ExecuteQuery(FeedIterator<Customer> feed)
    {
        List<Customer> results = new();
        while (feed.HasMoreResults)
        {
            var response = await feed.ReadNextAsync();
            foreach (Customer c in response)
            {
                results.Add(c);
            }
        }
        return results;
    }

    
}