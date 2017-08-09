import static org.junit.Assert.*;

/**
 * Created by cperkins on 11/7/15.
 */
public class WordNetTest {

    private WordNet wn;

    @org.junit.Before
    public void setUp() throws Exception {
        wn = new WordNet("synsets.txt", "hypernyms.txt");
    }

    @org.junit.Test
    public void testIsNoun() throws Exception {
        assertTrue(wn.isNoun("fart"));
        assertFalse(wn.isNoun("skjhkdf"));
    }

    @org.junit.Test
    public void testDistance() throws Exception {
        assertEquals(0, wn.distance("fart", "fart"));
    }

    @org.junit.Test
    public void testSap() throws Exception {

    }
}