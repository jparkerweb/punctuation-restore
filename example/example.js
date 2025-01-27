import PunctuationRestorer from '../punctuationRestore.js';

async function test() {
    try {
        const restorer = new PunctuationRestorer();
        
        // Test cases from the model card
        const testTexts = [
            "i woke up at 6 am and took the dog for a hike in the metacomet mountains we like to take morning adventures on the weekends despite being mid march it snowed overnight and into the morning here in connecticut it was snowier up in the mountains than in the farmington valley where i live",
            "george hw bush was the president of the us for 8 years",
            "i saw mr smith at the store he was shopping for a new lawn mower i suggested he get one of those new battery operated ones theyre so much quieter",
            "this is the first sentence to demonstrate how to create a text block with no punctuation it might feel awkward but i will keep going to meet the requirement right now i am trying to transition from one sentence to the next without using periods or commas i hope this approach isnt too challenging to read but that is the interesting part about no punctuation at all i wonder if this will truly feel like a run on paragraph or many just a crazy person"
        ];

        console.log('Processing test texts...');
        const results = await restorer.restore(testTexts);
        await restorer.cleanup();

        // console loge each result in results
        results.forEach(result => console.log(result));

    } catch (error) {
        console.error('Test failed:', error);
    }
}

test();
