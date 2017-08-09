import edu.princeton.cs.algs4.In;
import edu.princeton.cs.algs4.Out;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * Created by cperkins on 11/7/15.
 */
public class OutcastTest {

    @Test
    public void testOutcast() throws Exception {
        WordNet wn = new WordNet("synsets.txt", "hypernyms.txt");
        Outcast oc = new Outcast(wn);

        assertEquals("table",  oc.outcast(new String[]{"horse", "zebra", "cat", "bear", "table"}));
        assertEquals("bed",    oc.outcast(new String[]{"water", "soda", "bed", "orange_juice", "milk", "apple_juice", "tea", "coffee"}));
        assertEquals("potato", oc.outcast(new String[]{"apple", "pear", "peach", "banana", "lime", "lemon", "blueberry", "strawberry", "mango", "watermelon", "potato"}));
    }
}